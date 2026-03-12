#include <memory>
#include <sys/mman.h>
#include "util.h"
#include "buffers.hpp"
#include "rt_threads.hpp"
#include "network.hpp"

network::network (uint32_t threads, uint64_t period_ns)
  : m_num_threads (threads), m_period_ns (period_ns)
{
  /* Better to catch nonesense now than later on...  */
  assert (m_num_threads && period_ns);
}

network::~network ()
{
  kill ();
  /* We're responsible for the buffers allocated between layers,
     we have that BUFFER_RD for layer_i is BUFF_WR for layer_{i-1}
     and so forth...  */
  if (!m_layers.size () || !m_initialised)
    return;

  delete m_layers[0]->m_buffer_rd;
  for (uint32_t i = 0; i < m_layers.size (); i++)
    delete m_layers[i]->m_buffer_wr;
}

void
network::add_layer (std::unique_ptr<layer> l)
{
  assert (!m_initialised);
  if (!m_layers.empty ())
    assert (l->input_size () == m_layers.back ()->output_size ());

  m_layers.push_back (std::move (l));
}

void
network::initialise (std::vector<uint32_t> (*input_cb) (bool *),
		     void (*output_cb) (const std::vector<uint32_t> &))
{
  assert (!m_initialised && !m_layers.empty ());
  /* Profile any layer whose cost is undefined.  */
  for (auto &layer : m_layers)
    {
      if (layer->batch_cost () == COST_UNDEF)
	layer->profile_worstcase_batch ();
    }

  /* Distribute the layers across M_NUM_THREADS partitions.  */
  linear_partitioning ();

  /* Create the input thread and keep a copy of its address.  */
  auto input_thread
    = std::make_unique<input_rtt> (m_period_ns, input_cb,
				   m_layers.front ()->m_buffer_rd);
  m_input_thread = input_thread.get ();
  m_threads.push_back (std::move (input_thread));

  /* Create the output thread.  */
  m_threads.push_back
    (std::make_unique<output_rtt> (m_period_ns, output_cb,
				   m_layers.back ()->m_buffer_wr));

  m_initialised = true;
}

int
network::run ()
{
  assert (m_initialised);
  rts_checking_assert (!m_layers.empty () && m_input_thread != nullptr);

  int ret;
  pthread_barrier_t barrier;
  /* Lock memory for the entire process.  */
  ret = mlockall (MCL_CURRENT | MCL_FUTURE);
  if (ret)
    {
      debug_perror ("mlockall");
      debug_msg ("Failed to lock memory.\n");
      return ret;
    }

  /* We'll use a barrier to synchronise the start time of each thread.  */
  ret = pthread_barrier_init (&barrier, NULL, m_threads.size ());
  if (ret)
    {
      debug_perror ("pthread_barrier_init");
      debug_msg ("Failed to create synchonisation barrier.\n");
      return ret;
    }

  /* Create and run all threads.  */
  for (auto &thread : m_threads)
    {
      ret = thread->start (&barrier);
      if (ret)
	{
	  /* Bail!  */
	  kill ();
	  return ret;
	}
    }

  /* Wait for the input thread to die, then kill everything else.  */
  m_input_thread->join ();
  kill ();

  ret = pthread_barrier_destroy (&barrier);
  if (ret)
    {
      debug_perror ("pthread_barrier_destroy");
      debug_msg ("Failed to destroy synchonisation barrier.\n");
    }

#ifdef EN_PROFILE_NETWORK
  std::string stats = "";
  for (const auto &thread: m_threads)
    stats += thread->str_perf_metrics ();

  debug_dump ("{}\n", ";; Schematic");
  debug_dump ("{}\n", str_schematic_descr ());
  debug_dump ("{}\n", ";; Statistics");
  debug_dump ("{}", stats);
#endif

  return 0;
}

std::string
network::str_logical_descr (uint32_t level) const
{
  std::string temp = "";
  for (const auto &layer : m_layers)
    temp += "\n" + layer->str_descr (level + 1);

  std::string space = std::string (level, '\t');
  return std::format ("{}(network [{}\n{}])", space, temp, space);
}

std::string
network::str_schematic_descr (uint32_t level) const
{
  assert (m_initialised);

  std::string temp = "";
  for (const auto &thread : m_threads)
    temp += "\n" + thread->str_descr (level + 1);

  std::string space = std::string (level, '\t');
  return std::format ("{}(network [{}\n{}])", space, temp, space);
}

void
network::kill ()
{
  /* Killemall.  */
  for (auto &thread : m_threads)
    thread->kill ();
}

void
network::linear_partitioning ()
{
  rts_checking_assert (m_threads.empty () && m_layers.size ());

  /* Calculate the target load for each partition.  */
  uint64_t target = 0;
  for (const auto &layer : m_layers)
    target += layer->batch_cost () * layer->total_batches ();
  target /= m_num_threads;

  /* We'll allocate buffers as we go.  We know that the first buffer (that
     which receives data from the callback input and is read by the first
     layer) has exactly one writer.  */
  spikebuffer *buffer_prev = new spikebuffer ();
  buffer_prev->set_writers (1);

  /* Cost for current partition.  */
  uint64_t partition_cost = 0;
  /* Sublayers for the current partition.  */
  std::vector<sublayer> sublayers;
  for (auto &layer : m_layers)
    {
      /* The number of sublayers we've split this layer into.  */
      uint32_t l_slayers  = 0;
      uint64_t batch_cost = layer->batch_cost ();
      uint64_t batch_size = layer->batch_size ();

      /* Set sublayer info.  */
      uint32_t end   = 0;
      uint32_t begin = 0;

      /* Schedule the layer batch-by-batch.  */
      while (end != layer->output_size ())
	{
	  if (partition_cost + batch_cost / 2 <= target)
	    partition_cost += batch_cost;
	  else
	    {
	      if (end != 0)
		{
		  sublayers.emplace_back (layer.get (), begin, end);
		  l_slayers++;
		  /* The next sublayer starts from where this one ends.  */
		  begin = end;
		}
	      /* Record the current partition.  */
	      m_threads.push_back
		(std::make_unique<network_rtt> (m_period_ns, sublayers));
	      /* Reset SUBLAYERS and PARITION_COST.  */
	      sublayers.clear ();
	      partition_cost = batch_cost - (target - partition_cost);
	    }
	  end += batch_size;
	}
      /* We've reached the end of this layer, and by definition the current
	 sublayer.  */
      sublayers.emplace_back (layer.get (), begin, end);
      l_slayers++;

      /* Set BUFFER_RD for layer_i to BUFFER_WR of layer_{i-1}, but first,
	 tell that buffer how many readers it has.  */
      buffer_prev->set_readers (l_slayers);
      layer->m_buffer_rd = buffer_prev;

      /* Update BUFFER_PREV to that which this layer will write to.  */
      buffer_prev = new spikebuffer ();
      buffer_prev->set_writers (l_slayers);
      layer->m_buffer_wr = buffer_prev;
    }
  /* Record the final parition.  */
  m_threads.push_back
    (std::make_unique<network_rtt> (m_period_ns, sublayers));
  /* We know that the final buffer (that which receives data from the last
     layer and is read by the output callback) has exactly one reader.  */
  buffer_prev->set_readers (1);
}

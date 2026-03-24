#include <memory>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sys/mman.h>
#include "../include/util.h"
#include "../include/signal.hpp"
#include "../include/buffers.hpp"
#include "../include/threads.hpp"
#include "../include/network.hpp"

network::network (uint32_t threads, uint32_t period_us)
  : m_num_threads (threads), m_period_us (period_us)
{
  /* Better to catch nonesense now than later on...  */
  assert (m_num_threads && period_us);
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
network::initialise (input_thread::callback_type input_fn,
		     output_thread::callback_type output_fn)
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

  /* Create the input thread.  */
  m_threads.push_back
    (std::make_unique<input_thread> (m_period_us, input_fn,
				     m_layers.front ()->m_buffer_rd));
  /* Create the output thread.  */
  m_threads.push_back
    (std::make_unique<output_thread> (m_period_us, output_fn,
				      m_layers.back ()->m_buffer_wr));

  m_initialised = true;
}

int
network::run ()
{
  assert (m_initialised);
  rts_checking_assert (!m_layers.empty ());

#ifdef RTS_EN_LOCK_MEM
  /* Lock memory for the entire process.  */
  if (mlockall (MCL_CURRENT | MCL_FUTURE))
    {
      debug_perror ("mlockall");
      debug_msg ("Warn: failed to lock memory.\n");
    }
#endif

  /* This acts as a barrier to ensure that the threads begin their cyclic
     loops at approximately the same time.  */
  signal start_notification (m_threads.size ());

  /* Any thread may die at any time during the simulation; it may run out
     of work (e.g. the input thread) or experience a fatal error such as a
     timing violation.

     The loss of any thread is fatal for the whole simulation, so we should
     be notified when this happens and exit gracefully.  To do that, we'll
     use the same kind of signal/wait structure, where the main thread is
     the only waiter and all threads are potential signallers.  */
  signal exit_notification (1);

  /* Create and run all threads.  */
  for (auto it = m_threads.begin (); it != m_threads.end (); it++)
    {
      int ret = (*it)->start (&start_notification, &exit_notification);
      if (ret)
	{
	  debug_msg ("Failed to create threads\n.");
	  /* Break any waiting on START_NOTIFICATION, and join any threads
	     that started successfully.  */
	  start_notification.cancel ();
	  while (it != m_threads.begin ())
	    {
	      --it;
	      (*it)->kill_join ();
	    }

	  return ret;
	}
    }

  /* As discussed above, wait for any thread to die then kill and join
     all threads.  */
  exit_notification.wait ();
  kill ();

#ifdef RTS_EN_PROFILE_NETWORK
  /* Write the schematic.  */
  write_schematic (RTS_PERF_DIR"/schematic.txt");
  /* Write the performance data for each thread.  */
  for (const auto &thread: m_threads)
    thread->write_perf_metrics (std::format (RTS_PERF_DIR"/thread_{}_latencies",
					     thread->debug_id ()),
				std::format (RTS_PERF_DIR"/thread_{}_wakeups",
					     thread->debug_id ()));
#endif

  return 0;
}

void
network::kill ()
{
  /* Killemall.  */
  for (auto &thread : m_threads)
    thread->kill_join ();
}

void
network::linear_partitioning ()
{
  rts_checking_assert (m_threads.empty () && m_layers.size ());

  /* Calculate the target load for each partition.  */
  uint64_t target = 0;
  for (const auto &layer : m_layers)
    target += layer->batch_cost ()
      * (layer->output_size () / layer->batch_size ());
  target /= m_num_threads;

  /* We'll allocate buffers as we go.  We know that the first buffer (that
     which receives data from the callback input and is read by the first
     layer) has exactly one writer.  */
  spikebuffer *buffer_prev = new spikebuffer (m_layers[0]->input_size ());
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

      rts_checking_assert (batch_cost != COST_UNDEF);
      /* Schedule the layer batch-by-batch.  */
      while (end != layer->output_size ())
	{
	  if ((!partition_cost || m_threads.size () == (m_num_threads - 1))
	      || (partition_cost + batch_cost / 2 <= target))
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
		(std::make_unique<network_thread> (m_period_us, sublayers));
	      /* Reset SUBLAYERS and PARITION_COST.  */
	      sublayers.clear ();
	      if (partition_cost > target)
		/* Over-shoot.  */
		partition_cost = batch_cost + (partition_cost - target);
	      else
		/* Under-shoot.  */
		partition_cost = batch_cost + (target - partition_cost);
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
      layer->set_buffer_rd (buffer_prev);

      /* Update BUFFER_PREV to that which this layer will write to.  */
      buffer_prev = new spikebuffer (layer->output_size ());
      buffer_prev->set_writers (l_slayers);
      layer->set_buffer_wr (buffer_prev);
    }
  /* Record the final parition.  */
  m_threads.push_back
    (std::make_unique<network_thread> (m_period_us, sublayers));
  /* We know that the final buffer (that which receives data from the last
     layer and is read by the output callback) has exactly one reader.  */
  buffer_prev->set_readers (1);
}

std::vector<uint32_t>
network::generate_poisson_input (double rate_mhz) const
{
  rts_checking_assert (m_layers.size () != 0);
  /* A bernoulli approximation of each neuron as a poisson source.  */
  std::vector<uint32_t> spikes;
  std::uniform_real_distribution<double> dist (0, 1);

  double pred = rate_mhz * m_period_us;
  for (uint32_t i = 0; i < m_layers[0]->input_size (); i++)
    {
      if (dist (mersenne_twister ()) < pred)
	spikes.push_back (i);
    }

  return spikes;
}

std::vector<uint32_t>
network::inference (std::vector<uint32_t> spikes)
{
  /* The latter is just to prevent misuse.  */
  assert (m_layers.size () != 0 && !m_initialised);

#ifdef RTS_EN_PROFILE_NETWORK
  timespec start, end;
  clock_gettime (CLOCK_MONOTONIC, &start);
#endif
  for (auto &layer : m_layers)
    spikes = layer->forward (spikes);

#ifdef RTS_EN_PROFILE_NETWORK
  clock_gettime (CLOCK_MONOTONIC, &end);
  debug_dump ("latency: {} ns\n", (end.tv_sec - start.tv_sec) * 1E9
	      + end.tv_nsec - start.tv_nsec);
#endif

  return spikes;
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

int
network::write_schematic (const std::string &path) const
{
  std::ofstream file (path, std::ios::out | std::ios::trunc);
  if (!file.is_open ())
    {
      debug_msg ("Failed to write schematic to file: {}.\n", path);
      return -1;
    }

  file << str_schematic_descr (0);
  file.close ();

  return 0;
}

std::string
network::generate_performance_overview ()
{
  assert (m_initialised);

  std::string out = "";
#ifdef RTS_EN_PROFILE_NETWORK
  out += std::format (";; Schematic\n{}\n;; Latency info (all nanoseconds)\n",
		      str_schematic_descr (0));
  for (const auto &thread : m_threads)
    {
      const std::vector<uint64_t> &latencies = thread->latencies ();
      assert (!latencies.empty ());
      /* Maximum.  */
      uint64_t max = *std::max_element (latencies.begin (), latencies.end ());

      /* Mean.  */
      double mean
	= (double)std::accumulate (latencies.begin (), latencies.end (), 0.0)
	/ latencies.size ();

      /* Standard deviation.  */
      double sq_sum = 0.0;
      for (uint64_t x : latencies)
	sq_sum += (x - mean) * (x - mean);
      double stddev = std::sqrt (sq_sum / latencies.size ());

      out += std::format("thread {:<2} | mean: {:>10.2f} | max: {:>8} |"
			 "stddev: {:>10.2f}\n", thread->debug_id (), mean,
			 max, stddev);
    }
#else
  debug_msg ("Warn: metrics requested but profiling is disabled.\n");
#endif
  return out;
}

network::~network ()
{
  /* We're responsible for the buffers allocated between layers,
     we have that BUFFER_RD for layer_i is BUFF_WR for layer_{i-1}
     and so forth...  */
  if (!m_layers.size () || !m_initialised)
    return;

  delete m_layers[0]->m_buffer_rd;
  for (uint32_t i = 0; i < m_layers.size (); i++)
    delete m_layers[i]->m_buffer_wr;
}

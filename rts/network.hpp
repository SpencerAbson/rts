#ifndef  NETWORK_H
#define  NETWORK_H

#include <memory>
#include "buffers.hpp"
#include "layers/layer.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t period_ns=300000,
	   uint32_t profile_iters=100)
    : m_threads (threads), m_period_ns (period_ns),
      m_profile_iters (profile_iters), m_input_thread (period_ns),
      m_output_thread (period_ns)
  {
    /* Better to catch nonesense now than later on...  */
    assert (m_threads && period_ns && m_profile_iters);
  }

  void
  add_layer (std::unique_ptr<layer> l)
  {
    assert (!m_initialised);
    if (!m_layers.empty ())
      assert (l->input_size () == m_layers.back ()->output_size ());

    m_layers.push_back (std::move (l));
  }

  void
  initialise (std::vector<uint32_t> (*input_cb) (bool *),
	      void (*output_cb) (const std::vector<uint32_t> &))
  {
    assert (!m_initialised && !m_layers.empty ());
    /* Profile any layer whose cost is undefined.  */
    for (auto &layer : m_layers)
      {
	if (layer->batch_cost () == COST_UNDEF)
	  layer->profile_batch (m_profile_iters);
      }

    /* Distribute the layers across M_THREADS partitions.  */
    linear_partitioning ();

    /* Set the input thread callback/buffer.  */
    m_input_thread.set_callback (input_cb);
    m_input_thread.set_buffer (m_layers.front ()->m_buffer_rd);

    /* Likewise for the output thread.  */
    m_output_thread.set_callback (output_cb);
    m_output_thread.set_buffer (m_layers.back ()->m_buffer_wr);

    m_initialised = true;
  }

  int
  run ()
  {
    assert (m_initialised);
    rts_checking_assert (!m_layers.empty ());

    /* Select a start time conservatively far from now.  TODO: modify
       wait_rest_of_period to catch a potential violation here.  */
    timespec abs_start;
    clock_gettime (CLOCK_MONOTONIC, &abs_start);
    /* ??? 50ms.  */
    abs_start.tv_nsec += 5000000;
    handle_timespec_overflow (&abs_start);

    /* Start the network threads.  */
    int ret;
    for (auto &thread : m_network_threads)
      {
	ret = thread.start (abs_start);
	if (ret)
	  return ret;
      }
    /* ... I/O threads.  */
    ret = m_output_thread.start (abs_start);
    if (ret)
      return ret;

    ret = m_input_thread.start (abs_start);
    if (ret)
      return ret;

    /* Wait for the input thread to die, then kill everything else.  */
    m_input_thread.join ();
    kill ();

    return 0;
  }

  ~network ()
  {
    kill ();
    /* We're responsible for the buffers allocated between layers,
       we have that BUFFER_RD for layer_i is BUFF_WR for layer_{i-1}
       and so forth...  */
    if (!m_layers.size ())
      return;

    delete m_layers[0]->m_buffer_rd;
    for (uint32_t i = 0; i < m_layers.size (); i++)
      delete m_layers[i]->m_buffer_wr;
  }

private:

  void
  kill ()
  {
    /* Killemall.  */
    for (auto &thread : m_network_threads)
      thread.kill ();

    m_input_thread.kill ();
    m_output_thread.kill ();
  }

  void
  linear_partitioning ()
  {
    rts_checking_assert (m_network_threads.empty () && m_layers.size ());

    /* Calculate the target load for each partition.  */
    uint64_t target = 0;
    for (const auto &layer : m_layers)
      target += layer->batch_cost () * layer->total_batches ();
    target /= m_threads;

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
		m_network_threads.emplace_back (m_period_ns, sublayers);
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
    m_network_threads.emplace_back (m_period_ns, sublayers);
    /* We know that the final buffer (that which receives data from the last
       layer and is read by the output callback) has exactly one reader.  */
    buffer_prev->set_readers (1);
  }

  /* The number of threads used to parallelise the network.  */
  uint32_t m_threads;
  /* RT cyclic period.  */
  uint64_t m_period_ns;
  /* Number of profile measurements to take the arithmetic mean over.  */
  uint32_t m_profile_iters;
  /* The layers of a sequential spiking neural network (in order).  */
  std::vector<std::unique_ptr<layer>> m_layers;

  /* RT thread objects.  */
  input_rtt m_input_thread;
  output_rtt m_output_thread;
  std::vector<network_rtt> m_network_threads;

  bool m_initialised = false;
};

#endif //  NETWORK_H

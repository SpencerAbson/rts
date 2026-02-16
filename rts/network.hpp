#ifndef  NETWORK_H
#define  NETWORK_H

#include <atomic>
#include <memory>
#include <sched.h>
#include <pthread.h>
#include "buffers.hpp"
#include "layers/layer.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t period_ns=1000000,
	   uint32_t profile_iters=100)
    : m_threads (threads), m_period_ns (period_ns),
      m_profile_iters (profile_iters)
  {
    assert (m_threads != 0);
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
  initialise ()
  {
    assert (!m_initialised);
    /* Profile all layers whose cost is undefined.  */
    for (auto &layer : m_layers)
      {
	if (layer->batch_cost () == COST_UNDEF)
	  layer->profile_batch (m_profile_iters);
      }

    /* Distribute the layers across M_THREADS partitions.  */
    linear_partitioning ();
    m_initialised = true;
  }

  int
  run ()
  {
    assert (!m_alive && m_initialised);

    int ret;
    sched_param param;
    pthread_attr_t attr;

    /* Initialise RT pthread attributes.  */
    ret = pthread_attr_init (&attr);
    if (ret)
      {
	std::perror ("pthread_attr_init");
	return ret;
      }
    /* Set the scheduler policy of this pthread.  */
    ret = pthread_attr_setschedpolicy (&attr, SCHED_FIFO);
    if (ret)
      {
	std::perror ("pthread_attr_setschedpolicy");
	return ret;
      }
    /* Set the priority of this pthread.  */
    param.sched_priority = 80;
    ret = pthread_attr_setschedparam (&attr, &param);
    if (ret)
      {
	std::perror ("pthread_attr_setschedparam");
	return ret;
      }
    /* Use scheduling parameters of attr.  */
    ret = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
    if (ret)
      {
	std::perror ("pthread_attr_setinheritedsched");
	return ret;
      }

    /* Create each thread with ATTR...  */
  }

  void
  kill ()
  {
    if (!m_alive)
      return;

    m_alive = false;
    for (auto &worker : m_workers)
      {
	worker.alive = false;
	if (pthread_join (worker.id, NULL))
	    std::perror ("pthread_join");
      }
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
  linear_partitioning ()
  {
    rts_checking_assert (m_workers.empty ());

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
	sublayer slayer;
	slayer.begin = 0;
	slayer.end   = 0;
	slayer.l = layer.get ();

	/* Schedule the layer batch-by-batch.  */
	while (slayer.end != layer->output_size ())
	  {
	    if (partition_cost + batch_cost / 2 <= target)
	      partition_cost += batch_cost;
	    else
	      {
		if (slayer.end != 0)
		  {
		    sublayers.push_back (slayer);
		    l_slayers++;
		    /* The next sublayer starts from where this one ends.  */
		    slayer.begin = slayer.end;
		  }
		/* Record the current partition.  */
		m_workers.push_back (worker_info (sublayers, m_period_ns));
		/* Reset SUBLAYERS and PARITION_COST.  */
		sublayers.clear ();
		partition_cost = batch_cost - (target - partition_cost);
	      }
	    slayer.end += batch_size;
	  }
	/* We've reached the end of this layer, and by definition the current
	   sublayer.  */
	sublayers.push_back (slayer);
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
    m_workers.push_back (worker_info (sublayers, m_period_ns));
    /* We know that the final buffer (that which receives data from the last
       layer and is read by the output callback) has exactly one reader.  */
    buffer_prev->set_readers (1);
  }

  static void*
  worker_fn (void *arg)
  {
    worker_info *info = (worker_info *)arg;
    auto slayers = info->sublayers;
    uint64_t period_ns = info->period_ns;

    /* Wait for the coordinated start time.  */
    timespec timer = info->abs_start;
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);
    /* Loop until killed.  */
    while (info->alive)
      {
	for (sublayer &slayer : slayers)
	  slayer.l->run (slayer.begin, slayer.end);

	/* Wait for the remaining period.  */
	timer.tv_nsec += period_ns;
	handle_timespec_overflow (timer);
	clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);
      }

    return NULL;
  }

  struct sublayer
  {
    layer *l;
    uint32_t begin;
    uint32_t end;
  };
  /* The argument to each pthread.  */
  struct worker_info
  {
    /* Workload.  */
    std::vector<sublayer> sublayers;
    /* Copy of network-wide period parameter.  */
    uint64_t period_ns;
    /* Absolute start time.  */
    timespec abs_start;
    /* Killswitch.  */
    bool alive;
    /* Thread ID.  */
    pthread_t id;

    worker_info (std::vector<sublayer> slayers, uint64_t period)
      : sublayers (slayers), period_ns (period), alive (false)
    {}
  };

  uint32_t m_threads;
  uint64_t m_period_ns;
  uint32_t m_profile_iters;

  std::vector<worker_info> m_workers;
  std::vector<std::unique_ptr<layer>> m_layers;

  bool m_alive = false;
  bool m_initialised = false;
};

#endif //  NETWORK_H

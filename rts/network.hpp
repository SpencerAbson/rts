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
  initialise (std::vector<uint32_t> (*input_cb) (void),
	      void (*output_cb) (const std::vector<uint32_t> &))
  {
    assert (!m_initialised && !m_layers.empty ());
    /* Profile all layers whose cost is undefined.  */
    for (auto &layer : m_layers)
      {
	if (layer->batch_cost () == COST_UNDEF)
	  layer->profile_batch (m_profile_iters);
      }

    /* Distribute the layers across M_THREADS partitions.  */
    linear_partitioning ();

    /* Setup the input and output workers.  The former writes to the first
       layer's BUFFER_RD.  */
    m_input_worker.cb	= input_cb;
    m_input_worker.period_ns = m_period_ns;
    m_input_worker.buffer    = (*m_layers.begin ())->m_buffer_rd;

    /* The latter reads from the last layer's BUFFER_WR.  */
    m_output_worker.cb	= output_cb;
    m_output_worker.period_ns = m_period_ns;
    m_output_worker.buffer    = (*m_layers.end ())->m_buffer_wr;

    m_initialised = true;
  }

  int
  run ()
  {
    assert (!m_alive && m_initialised);
    rts_checking_assert (!m_layers.empty ());

    timespec start;
    pthread_attr_t attr;
    /* The same attributes are currently shared by all RT threads.  */
    int ret = initialise_rt_attr (&attr, 80);
    if (ret)
      {
	debug_printf ("Failed to initialise RT pthread attributes.\n");
	return ret;
      }

    /* Coordinate a start time for all threads.  */
    clock_gettime (CLOCK_MONOTONIC, &start);
    /* ??? 50ms should be conservative enough.  */
    start.tv_nsec += 50000000;
    handle_timespec_overflow (&start);

    /* Create the worker threads.  */
    for (auto &worker : m_layer_workers)
      {
	worker.abs_start = start;
	ret = pthread_create (&worker.id, &attr, layer_worker_fn,
			      (void *)&worker);

	if (ret)
	  {
	    kill ();
	    debug_perror ("pthread_create");
	    return ret;
	  }
      }

    /* Similarly, for the I/O threads.  */
    m_input_worker.abs_start = start;
    ret = pthread_create (&m_input_worker.id, &attr, input_worker_fn,
			  (void *)&m_input_worker);
    if (ret)
      {
	kill ();
	debug_perror ("pthread_create");
	return ret;
      }

    m_output_worker.abs_start = start;
    ret = pthread_create (&m_output_worker.id, &attr, output_worker_fn,
			  (void *)&m_output_worker);
    if (ret)
      {
	kill ();
	debug_perror ("pthread_create");
	return ret;
      }

    return 0;
  }

  void
  kill ()
  {
    rts_checking_assert (m_initialised);
    /* Firstly, kill the I/O threads if they aren't dead already.  */
    if (kill_worker (&m_input_worker))
      debug_printf ("Failed to kill the input worker.");

    if (kill_worker (&m_output_worker))
      debug_printf ("Failed to kill the output worker.");

    /* Next, the layer threads.  */
    for (auto &worker : m_layer_workers)
      {
	if (kill_worker (&worker))
	  debug_printf ("Failed to kill a layer worker.");
      }

    m_alive = false;
  }

  ~network ()
  {
    if (!m_initialised)
      return;
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
    rts_checking_assert (m_layer_workers.empty ());

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
		m_layer_workers.emplace_back (sublayers, m_period_ns);
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
    m_layer_workers.emplace_back (sublayers, m_period_ns);
    /* We know that the final buffer (that which receives data from the last
       layer and is read by the output callback) has exactly one reader.  */
    buffer_prev->set_readers (1);
  }

  struct sublayer
  {
    layer *l;
    uint32_t begin;
    uint32_t end;

    sublayer (layer *l, uint32_t begin, uint32_t end)
      : l (l), begin (begin), end (end)
    {}
  };
  /* The base of the argument to each pthread.  */
  struct worker_info
  {
    /* Copy of network-wide period parameter.  */
    uint64_t period_ns;
    /* Absolute start time.  */
    timespec abs_start;
    /* Killswitch.  */
    std::atomic<bool> alive{false};
    /* Thread ID.  */
    pthread_t id;

    worker_info () = default;

    worker_info (uint64_t period)
      : period_ns (period)
    {}

    worker_info (const worker_info &other)
      : period_ns (other.period_ns),
	alive (other.alive.load (std::memory_order_seq_cst))
    {}
  };
  /* The argument to each layer pthread.  */
  struct layer_worker : worker_info
  {
    /* Workload.  */
    std::vector<sublayer> sublayers;

    layer_worker (std::vector<sublayer> slayers, uint64_t period)
      : worker_info (period), sublayers (slayers)
    {}
  };
  /* The argument to the input pthread.  */
  struct input_worker : worker_info
  {
    /* Written to by us only, read by threads running the first layer.  */
    spikebuffer *buffer;
    /* Callback.  */
    std::vector<uint32_t> (*cb) (void);
  };
  /* The argument to the output pthread.  */
  struct output_worker : worker_info
  {
    /* Written to by threads running the last layer, read by us only.  */
    spikebuffer *buffer;
    /* Callback.  */
    void (*cb) (const std::vector<uint32_t> &);
  };

  int
  initialise_rt_attr (pthread_attr_t *attr, int32_t priority)
  {
    sched_param param;
    int ret = pthread_attr_init (attr);
    if (ret)
      {
	debug_perror ("pthread_attr_init");
	return ret;
      }
    /* Set scheduler policy and priority of the pthread.  */
    ret = pthread_attr_setschedpolicy (attr, SCHED_FIFO);
    if (ret)
      {
	debug_perror ("pthread_attr_setschedpolicy");
	return ret;
      }
    param.sched_priority = priority;
    ret = pthread_attr_setschedparam (attr, &param);
    if (ret)
      {
	debug_perror ("pthread_attr_setschedparam");
	return ret;
      }
    /* Use scheduling parameters of attr.  */
    ret = pthread_attr_setinheritsched (attr, PTHREAD_EXPLICIT_SCHED);
    if (ret)
      {
	debug_perror ("pthread_attr_setinheritedsched");
	return ret;
      }

    return 0;
  }

  static int
  kill_worker (worker_info *worker)
  {
    int ret = 0;
    if (worker->alive.load (std::memory_order_acquire))
      {
	worker->alive.store (false, std::memory_order_release);
	ret = pthread_join (worker->id, NULL);

	if (ret)
	  debug_perror ("pthread_join");
      }

    return ret;
  }

  /* Sleep until the clock reaches TS + PERIOD_NS.  NOTE:  We ought to be
     checking whether this time has already been reached...  */
  static void
  wait_rest_of_period (timespec *ts, uint64_t period_ns)
  {
    ts->tv_nsec += period_ns;
    handle_timespec_overflow (ts);
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, ts, NULL);
  }

  /* Function executed by layer RT threads.  */
  static void*
  layer_worker_fn (void *arg)
  {
    layer_worker *info = (layer_worker *)arg;
    info->alive.store (true, std::memory_order_release);

    timespec timer = info->abs_start;
    uint64_t period_ns = info->period_ns;
    std::vector<sublayer> slayers = info->sublayers;

    /* Sleep until the absolute start time.  */
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);

    /* Main RT cyclic task for layer workers.  */
    while (info->alive.load (std::memory_order_acquire))
      {
	for (sublayer &slayer : slayers)
	  slayer.l->run (slayer.begin, slayer.end);

	wait_rest_of_period (&timer, period_ns);
      }

    return NULL;
  }

  /* Funcion executed by the input RT thread.  */
  static void*
  input_worker_fn (void *arg)
  {
    input_worker *info = (input_worker *)arg;
    info->alive.store (true, std::memory_order_release);

    timespec timer = info->abs_start;
    uint64_t period_ns = info->period_ns;
    spikebuffer *buffer = info->buffer;
    std::vector<uint32_t> (*cb) (void) = info->cb;

    /* Sleep until the absolute start time.  */
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);

    /* Main RT cyclic task for the input worker.  */
    while (info->alive.load (std::memory_order_acquire))
      {
	buffer->write (cb ());
	wait_rest_of_period (&timer, period_ns);
      }

    return NULL;
  }

   /* Funcion executed by the input RT thread.  */
  static void*
  output_worker_fn (void *arg)
  {
    output_worker *info = (output_worker *)arg;
    info->alive.store (true, std::memory_order_release);

    timespec timer = info->abs_start;
    uint64_t period_ns = info->period_ns;
    spikebuffer *buffer = info->buffer;
    void (*cb) (const std::vector<uint32_t> &) = info->cb;

    /* Sleep until the absolute start time.  */
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);

    /* Main RT cyclic task for the output worker.  */
    while (info->alive.load (std::memory_order_acquire))
      {
	cb (buffer->read ());
	wait_rest_of_period (&timer, period_ns);
      }

    return NULL;
  }

  uint32_t m_threads;
  uint64_t m_period_ns;
  uint32_t m_profile_iters;

  input_worker m_input_worker;
  output_worker m_output_worker;

  std::vector<layer_worker> m_layer_workers;
  std::vector<std::unique_ptr<layer>> m_layers;

  bool m_alive = false;
  bool m_initialised = false;
};

#endif //  NETWORK_H

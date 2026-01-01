#ifndef  NETWORK_H
#define  NETWORK_H

#include "layers/layer.hpp"
#include "buffers.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t sleep_ns=80000,
	   uint32_t profile_iters=100)
    :  m_sleep_ns (sleep_ns), m_max_threads (threads),
       m_profile_iters (profile_iters)
  {}

  void
  add_layer (layer *l)
  {
    assert (!m_initialised);
    if (!m_layers.empty ())
      assert (l->input_size () == m_layers.back ()->output_size ()
	      && "Mismatched layer dimensions.");

    m_layers.push_back (l);
  }

  bool
  alive ()
  {
    return m_alive.load (std::memory_order_acquire);
  }

  bool
  write (const std::vector<uint32_t> &vec)
  {
    assert (alive ());
    return m_stages[0]->buffer_rd->write (vec);
  }

  void
  write_blocking (const std::vector<uint32_t> &vec)
  {
    struct timespec sleep;
    sleep.tv_sec = 0;
    sleep.tv_nsec = m_sleep_ns;

    while (!(write (vec)))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
  }

  bool
  read (std::vector<uint32_t> &vec)
  {
    assert (m_initialised);
    return (*--m_stages.end ())->buffer_wr->latch (vec);
  }

  void
  read_blocking (std::vector<uint32_t> &vec)
  {
    struct timespec sleep;
    sleep.tv_sec = 0;
    sleep.tv_nsec = m_sleep_ns;

    while (!(read (vec)) && alive ())
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
  }

  void
  init ()
  {
    assert (!alive () && !m_initialised);
    initialise ();
  }

  void
  run ()
  {
    assert (!alive () && m_initialised);
    m_alive.store (true, std::memory_order_release);

    int res;
    for (auto stage : m_stages)
      {
	res = pthread_create (&stage->id, NULL, stage_worker, stage);
	assert (res == 0 && "pthread creation failed.");
      }
  }

  void
  kill ()
  {
    m_alive.store (false, std::memory_order_release);

    int res;
    for (auto stage : m_stages)
      {
	stage->alive.store (false, std::memory_order_release);
	res = pthread_join (stage->id, NULL);
	assert (res == 0 && "pthread join failed.");
      }
  }

  ~network ()
  {
    if (!m_initialised)
      return;
    rts_checking_assert (m_stages.size ());
    /* For i=1..n, buffer_rd_{i} = buffer_wr_{i-1}, so we should free
       buffer_wr only for all but the first stage.  */
    delete m_stages[0]->buffer_rd;
    for (uint32_t i = 0; i < m_stages.size (); i++)
      {
	delete m_stages[i]->buffer_wr;
	delete m_stages[i];
      }
  }

private:

  void
  linear_partitioning ()
  {
    /* Greedily partition the layers into up to M_MAX_THREADS stages, such
       that the load inbalance is minimised, with the constraint that the
       layers in each partition must have been contiguous in the original
       list.

       This is known as 'linear partitioning' in literature.  */
    rts_checking_assert (m_stages.empty ());

    /* Calculate the optimal load for each stage.  */
    uint64_t target = 0;
    for (auto layer : m_layers)
	target += layer->timestep_cost ();
    target /= m_max_threads;

    /* We'll allocate stages and buffers as we go.  With the current
       parallelism scheme, each buffer has a single producer and a
       single consumer, so all buffers can be SPSC_BUFFER.  */
    stage_info *stage = nullptr;
    buffer<uint32_t> *buffer_prev
      = new spsc_buffer<uint32_t> ();

    uint64_t layer_cost, partition_cost = 0;
    for (auto layer : m_layers)
      {
	layer_cost = layer->timestep_cost ();
	if ((stage && partition_cost + layer_cost / 2 <= target)
	    || m_stages.size () >= m_max_threads)
	  {
	    stage->layers.push_back (layer);
	    partition_cost += layer_cost;
	  }
	else
	  {
	    if (!stage)
	      /* Initial stage, not the end of a partition.  */
	      partition_cost = layer_cost;
	    else
	      /* Offset the next cost by current error.  */
	      partition_cost = layer_cost - (target - partition_cost);

	    stage = new stage_info ();
	    stage->layers.push_back (layer);
	    stage->buffer_rd = buffer_prev;
	    stage->buffer_wr = new spsc_buffer<uint32_t> ();
	    stage->sleep_ns  = m_sleep_ns;
	    stage->alive     = true;
	    /* Set buffer_rd for stage_{i+1} to buffer_wr_i.  */
	    buffer_prev      = stage->buffer_wr;
	    m_stages.push_back (stage);
	  }
      }
  }

  void
  initialise ()
  {
    assert (m_max_threads != 0 && m_layers.size () >= m_max_threads);
    /* Profile all layers whose cost is undefined.  */
    for (auto layer : m_layers)
      {
	if (layer->timestep_cost () == COST_UNDEF)
	  layer->profile (m_profile_iters);
      }

    /* Distribute layers across M_MAX_THREADS stages.  */
    linear_partitioning ();
    m_initialised = true;
  }

  static void*
  stage_worker (void *arg)
  {
    stage_info *s_info = (stage_info *)arg;
    /* We don't want to be dereferencing these every cycle, so latch
       them here.  */
    buffer<uint32_t> *buffer_rd = s_info->buffer_rd;
    buffer<uint32_t> *buffer_wr = s_info->buffer_wr;
    std::vector<layer *> layers = s_info->layers;

    /* For nanosleeping.  */
    struct timespec sleep;
    sleep.tv_sec  = 0;
    sleep.tv_nsec = s_info->sleep_ns;

    std::vector<uint32_t> result;
    while (s_info->alive.load (std::memory_order_acquire))
      {
	/* Wait for the input to be ready.  */
	while (!buffer_rd->latch (result))
	  {
	    clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
	    /* Check if we've been killed while waiting.  */
	    if (!s_info->alive.load (std::memory_order_acquire))
	      return NULL;
	  }

	for (auto layer : layers)
	  result = layer->timestep (result);

	/* Wait until output can be written, and similarly check
	   that we haven't been killed.  */
	while (s_info->alive.load (std::memory_order_acquire)
	       && !buffer_wr->write (result))
	  clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
      }

    return NULL;
  }

  std::atomic<bool> m_alive;
  bool m_initialised;
  uint64_t m_sleep_ns;
  uint32_t m_max_threads;
  uint32_t m_profile_iters;

  struct stage_info
  {
    /* layer(s) run during this stage.  */
    std::vector<layer *> layers;
    /* Input buffer for first layer of this stage.  */
    buffer<uint32_t> *buffer_rd;
    /* Output buffer for the last layer this stage.  */
    buffer<uint32_t> *buffer_wr;
    /* Copy of network-wide sleep paramter.  */
    uint64_t sleep_ns;
    /* Thread ID.  */
    pthread_t id;
    /* Killswitch.  */
    std::atomic<bool> alive;
  };

  std::vector<layer *> m_layers;
  std::vector<stage_info *> m_stages;
};

#endif //  NETWORK_H

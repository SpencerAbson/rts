#ifndef  NETWORK_H
#define  NETWORK_H

#include "layers/layer.hpp"
#include "buffers.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t sleep_ns=80000)
    :  m_sleep_ns (sleep_ns), m_max_threads (threads)
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

  bool alive () { return m_alive; }

  bool
  write (const std::vector<uint32_t> &vec)
  {
    assert (m_alive);
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

    while (!(read (vec)))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
  }

  void
  init ()
  {
    assert (!m_alive && !m_initialised);
    initialise ();
  }

  void
  run ()
  {
    assert (!m_alive && m_initialised);
    m_alive = true;

    /* Dispatch each stage.  */
    pthread_t thread;
    for (stage_info *s : m_stages)
      pthread_create (&thread, NULL, stage_worker, s);
  }

  void
  kill ()
  {
    m_alive = false;
    for (auto stage : m_stages)
      stage->alive = false;
  }

  ~network ()
  {
    if (m_initialised)
      {
	rts_checking_assert (m_stages.size ());
	delete m_stages[0]->buffer_rd;
	delete m_stages[0]->buffer_wr;
	delete m_stages[0];
	/* Subsequent stages should only free their write buffer,
	   since buffer_rd_{i+1} = buffer_wr_i.  */
	for (uint32_t i = 1; i < m_stages.size (); i++)
	  {
	    delete m_stages[i]->buffer_wr;
	    delete m_stages[i];
	  }
      }
  }

private:

  void
  initialise ()
  {
    assert (m_max_threads != 0 && m_layers.size () >= m_max_threads);
    /* FORNOW: Each thread/stage processes a set of successive layers
       as defined by the network.  This leaves a lot of opportunity
       untapped, such as breaking down individual layers arbitrarily,
       but is a simple way to pipeline an SNN.

       A hacky greedy load-balancing algorithm that operates within
       these constraints is given below.  */
    uint32_t target = 0;
    for (layer *l : m_layers)
      target += l->timestep_cost ();
    /* The optimal load for each stage.  */
    target /= m_max_threads;

    /* Distribute the layers across M_MAX_THREADS stages.  */
    buffer<uint32_t> *buffer_prev = new spsc_buffer<uint32_t> ();
    stage_info *stage = nullptr;
    uint32_t l_cost, s_cost = 0;
    for (layer *l : m_layers)
      {
	l_cost = l->timestep_cost ();
	/* If this layer improves the balance for the current stage,
	   or if we've run out threads to create a new one...   */
	if (stage
	    && (ABSDIFF (s_cost + l_cost, target) < ABSDIFF (s_cost, target)
		|| m_stages.size () >= m_max_threads))
	  {
	    /* Add L to the current stage.  */
	    s_cost += l_cost;
	    stage->layers.push_back (l);
	  }
	else
	  {
	    /* Otherwise, create a new stage/thread.  */
	    stage = new stage_info ();
	    stage->layers.push_back (l);
	    stage->buffer_rd = buffer_prev;
	    stage->buffer_wr = new spsc_buffer<uint32_t> ();
	    stage->sleep_ns  = m_sleep_ns;
	    stage->alive     = true;
	    /* Set buffer_rd for stage_{i+1} to buffer_wr_i.  */
	    buffer_prev      = stage->buffer_wr;
	    m_stages.push_back (stage);
	    s_cost = 0;
	  }
      }

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
    while (s_info->alive)
      {
	/* Wait for input to be ready.  */
	while (!(buffer_rd->latch (result)) && s_info->alive)
	  clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);

	/* Run the layers covered by this stage.  */
	for (auto layer : layers)
	  result = layer->timestep (result);

	/* Wait for output to be latchable?  */
	while (!(buffer_wr->write (result)) && s_info->alive)
	  clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
      }

    return NULL;
  }

  bool m_alive;
  bool m_initialised;
  uint64_t m_sleep_ns;
  uint32_t m_max_threads;

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
    /* Killswitch.  */
    bool alive;
  };

  std::vector<layer *> m_layers;
  std::vector<stage_info *> m_stages;
};

#endif //  NETWORK_H

#ifndef  NETWORK_H
#define  NETWORK_H

#include <atomic>
#include "layers/layer.hpp"
#include "buffers.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t period_ns=1000000,
	   uint32_t profile_iters=100)
    : m_threads (threads), m_period_ns (period_ns),
      m_profile_iters (profile_iters) {}

  void
  add_layer (layer *l)
  {
    assert (!m_initialised);
    if (!m_layers.empty ())
      assert (l->input_size () == m_layers.back ()->output_size ());

    m_layers.push_back (l);
  }

  void
  initialise ()
  {
    assert (!m_initialised);

    /* Profile and scehedule... blah blah blah... */
  }

  ~network ()
  {
    if (!m_initialised)
      return;
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

  static void
  increment_time (struct timespec *time, uint64_t period_ns)
  {
    time->tv_nsec += period_ns;
    while (time->tv_nsec >= 1000000000)
      {
	/* timespec nsec overflow.  */
	time->tv_sec++;
	time->tv_nsec -= 1000000000;
      }
  }

  static void*
  worker_fn (void *arg)
  {
    worker_info *info = (worker_info *)arg;
    /* Timing.  */
    uint64_t period_ns    = info->period_ns;
    struct timespec timer = info->abs_start;
    /* workload.  */
    std::vector<sublayer> slayers = info->sublayers;

    /* Wait for the agreed start time.  */
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &timer, NULL);
    /* Loop until killed.  */
    while (info->alive.load (std::memory_order_acquire))
      {
	for (sublayer &slayer : slayers)
	  slayer.l->run (slayer.begin, slayer.end);

	/* Wait rest of period.  */
	increment_time (&timer, period_ns);
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
    struct timespec abs_start;
    /* Killswitch.  */
    std::atomic<bool> alive;
    /* Thread ID.  */
    pthread_t id;
  };

  uint32_t m_threads;
  uint64_t m_period_ns;
  uint32_t m_profile_iters;

  std::vector<layer *> m_layers;
  std::vector<worker_info> m_workers;
  /* Network state.  */
  bool m_initialised = false;
  std::atomic<bool> m_alive = false;
};

#endif //  NETWORK_H

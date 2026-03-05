#include <sched.h>
#include <pthread.h>
#include "util.h"
#include "rt_threads.hpp"

rt_thread::rt_thread (uint64_t period_ns, int priority)
  : m_period_ns (period_ns), m_priority (priority)
{}

int
rt_thread::start (pthread_barrier_t *barrier)
{
  pthread_attr_t attr;
  int ret = pthread_attr_init (&attr);
  if (ret)
    {
      debug_perror ("pthread_attr_init");
      debug_printf ("Failed to initialise pthread attrs.\n");
      return ret;
    }

  /* Set scheduler policy.  */
  ret = pthread_attr_setschedpolicy (&attr, SCHED_FIFO);
  if (ret)
    {
      debug_perror ("pthread_attr_setschedpolicy");
      debug_printf ("Failed to set pthread scheduler policy.\n");
      return ret;
    }

  /* Set scheduler priority.  */
  struct sched_param param;
  param.sched_priority = m_priority;
  ret = pthread_attr_setschedparam (&attr, &param);
  if (ret)
    {
      debug_perror ("pthread_attr_setschedparam");
      debug_printf ("Failed to set pthread scheduler priority.\n");
      return ret;
    }

  /* Ensure pthread_create uses our ATTR rather than those of the
     parent thread.  */
  ret = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
  if (ret)
    {
      debug_perror ("pthread_attr_setinheritsched");
      debug_printf ("Failed to set pthread scheduler attributes.\n");
      return ret;
    }

  /* Copy the barrier over.  */
  m_barrier = barrier;
  /* Create the thread.  */
  ret = pthread_create (&m_id, &attr, runner, (void *)this);
  return 0;
}

int
rt_thread::join ()
{
  int ret = pthread_join (m_id, NULL);
  if (ret)
    {
      debug_perror ("pthread_join");
      debug_printf ("Failed to join pthread.\n");
    }

  return ret;
}

int
rt_thread::kill ()
{
  int ret = 0;
  if (m_alive.load (std::memory_order_acquire))
    {
      m_alive.store (false, std::memory_order_relaxed);
      ret = join ();
    }
  if (ret)
    debug_printf ("Failed to properly kill pthread.\n");

  return ret;
}

void
rt_thread::complete_period ()
{
#ifdef EN_PROFILE_NETWORK
  timespec end;
  clock_gettime (CLOCK_MONOTONIC, &end);
  /* Record latency stats.  */
  uint64_t latency = (end.tv_sec - m_timer.tv_sec) * 1E9
    + end.tv_nsec - m_timer.tv_nsec;

  if (latency > m_max_latency_ns)
    m_max_latency_ns = latency;
  if (latency < m_min_latency_ns)
    m_min_latency_ns = latency;

  m_total_latency_ns += latency;
  m_total_cycles++;
#endif

  m_timer.tv_nsec += m_period_ns;
  handle_timespec_overflow (&m_timer);

  clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &m_timer, NULL);
}


network_rtt::network_rtt (uint64_t period_ns, std::vector<sublayer> slayers,
			  int priority)
  : rt_thread (period_ns, priority), m_sublayers (slayers)
{}

void
network_rtt::run ()
{
  while (m_alive.load (std::memory_order_relaxed))
    {
      for (sublayer &slayer : m_sublayers)
	slayer.l->run (slayer.begin, slayer.end);

      complete_period ();
    }
}


input_rtt::input_rtt (uint64_t period_ns, std::vector<uint32_t> (*cb) (bool *),
		      spikebuffer *buff, int priority)
  : rt_thread (period_ns, priority), m_buffer (buff), m_cb (cb)
{}

void
input_rtt::run ()
{
  bool killed = false;
  while (!killed && m_alive.load (std::memory_order_relaxed))
    {
      m_buffer->write (m_cb (&killed));
      complete_period ();
    }

  /* Propagate death.  */
  m_alive.store (false, std::memory_order_relaxed);
}


output_rtt::output_rtt (uint64_t period_ns,
			void (*cb) (const std::vector<uint32_t> &),
			spikebuffer *buff, int priority)
  : rt_thread (period_ns, priority), m_buffer (buff), m_cb (cb)
{}

void
output_rtt::run ()
{
  while (m_alive.load (std::memory_order_relaxed))
    {
      m_cb (m_buffer->read ());
      complete_period ();
    }
}

#ifndef RT_THREADS_H_
#define RT_THREADS_H_

#include <sched.h>
#include <pthread.h>
#include <atomic>
#include "util.h"
#include "layers/layer.hpp"


class rt_thread
{
public:

  rt_thread (uint64_t period_ns, int policy=SCHED_FIFO, int priority=80)
    : m_period_ns (period_ns), m_policy (policy), m_priority (priority)
  {};

  /* RT cyclic task implementation.  */
  virtual void run () = 0;

  int
  start (timespec start)
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
    ret = pthread_attr_setschedpolicy (&attr, m_policy);
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

    /* Copy the start time.  */
    m_timer = start;
    /* Create the thread.  */
    ret = pthread_create (&m_id, &attr, runner, (void *)this);
    return 0;
  }

  int
  join ()
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
  kill ()
  {
    int ret = 0;
    if (m_alive.load (std::memory_order_acquire))
      {
	m_alive.store (false, std::memory_order_release);
	ret = join ();
      }
    if (ret)
      debug_printf ("Failed to properly kill pthread.\n");

    return ret;
  }

  static void*
  runner (void *arg)
  {
    rt_thread *thread = (rt_thread *)arg;
    /* Set state to alive.  */
    thread->m_alive.store (true, std::memory_order_relaxed);

    /* Sleep until the absolute start time.  */
    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &thread->m_timer, NULL);

    /* Run!  */
    thread->run ();
    return NULL;
  }

  void
  wait_rest_of_period ()
  {
    m_timer.tv_nsec += m_period_ns;
    handle_timespec_overflow (&m_timer);

    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &m_timer, NULL);
  }

  /* Copy of network-wide period parameter.  */
  uint64_t m_period_ns;
  /* Timer, initially the absolute start time.  */
  timespec m_timer;
  /* Killswitch.  */
  std::atomic<bool> m_alive {false};
  /* pthread API info.  */
  pthread_t m_id;
  int m_policy;
  int m_priority;

  /* I would leave this as default but the compiler understandably deletes
     the default copy constructor on atomics...   */
  rt_thread (const rt_thread& other)
    : m_period_ns (other.m_period_ns), m_timer (other.m_timer),
      m_alive (other.m_alive.load ()), m_id (other.m_id),
      m_policy (other.m_policy), m_priority (other.m_priority)
  {}
};


struct sublayer
{
  layer *l;
  uint32_t begin;
  uint32_t end;

  sublayer (layer *l, uint32_t begin, uint32_t end)
    : l (l), begin (begin), end (end)
  {}
};

class network_rtt : public rt_thread
{
public:

  network_rtt (uint64_t period_ns, std::vector<sublayer> slayers,
	       int policy=SCHED_FIFO, int priority=80)
    : rt_thread (period_ns, policy, priority), m_sublayers (slayers)
  {}

  void
  run ()
  {
    while (m_alive.load (std::memory_order_acquire))
      {
	for (sublayer &slayer : m_sublayers)
	  slayer.l->run (slayer.begin, slayer.end);

	wait_rest_of_period ();
      }
  }

private:
  /* Workload.  */
  std::vector<sublayer> m_sublayers;
};


class input_rtt : public rt_thread
{
public:

  input_rtt (uint64_t period_ns, int policy=SCHED_FIFO, int priority=80)
    : rt_thread (period_ns, policy, priority)
  {}

  void
  set_callback (std::vector<uint32_t> (*cb) (bool *))
  {
    m_cb = cb;
  }

  void
  set_buffer (spikebuffer *buff)
  {
    m_buffer = buff;
  }

  void
  run ()
  {
    rts_checking_assert (m_cb != nullptr && m_buffer != nullptr);

    bool killed = false;
    while (!killed && m_alive.load (std::memory_order_acquire))
      {
	m_buffer->write (m_cb (&killed));
	wait_rest_of_period ();
      }

    /* Propagate death.  */
    m_alive.store (false, std::memory_order_release);
  }

private:
  /* Written to by us only, read by threads running the first layer.  */
  spikebuffer *m_buffer = nullptr;
  /* Callback.  */
  std::vector<uint32_t> (*m_cb) (bool *) = nullptr;
};

class output_rtt : public rt_thread
{
public:

  output_rtt (uint64_t period_ns, int policy=SCHED_FIFO, int priority=80)
    : rt_thread (period_ns, policy, priority)
  {}

  void
  set_callback (void (*cb) (const std::vector<uint32_t> &))
  {
    m_cb = cb;
  }

  void
  set_buffer (spikebuffer *buff)
  {
    m_buffer = buff;
  }

  void
  run ()
  {
    rts_checking_assert (m_cb != nullptr && m_buffer != nullptr);

    while (m_alive.load (std::memory_order_acquire))
      {
	m_cb (m_buffer->read ());
	wait_rest_of_period ();
      }
  }

private:
  /* Written to by threads running the last layer, read by us only.  */
  spikebuffer *m_buffer = nullptr;
  /* Callback.  */
  void (*m_cb) (const std::vector<uint32_t> &) = nullptr;
};

#endif // RT_THREADS_H_

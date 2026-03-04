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

  rt_thread (uint64_t period_ns, int priority=80)
    : m_period_ns (period_ns), m_priority (priority)
  {};

  /* RT cyclic task implementation.  */
  virtual void run () = 0;

  /* Initialise and run, returning 0 on success and a pthread error
     code otherwise.  */
  int
  start (pthread_barrier_t *barrier)
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

  /* Wrapper on pthread_join.  */
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

  /* Kill and join, returning 0 on success and a pthread error
     code otherwise.  */
  int
  kill ()
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

  static void*
  runner (void *arg)
  {
    rt_thread *thread = (rt_thread *)arg;
    rts_checking_assert (thread->m_barrier != nullptr);

    /* Set state to alive.  */
    thread->m_alive.store (true, std::memory_order_relaxed);
    /* Sleep until everyone else is ready.  */
    pthread_barrier_wait (thread->m_barrier);
    /* Set the timer.  */
    clock_gettime (CLOCK_MONOTONIC, &thread->m_timer);

    /* Run!  */
    thread->run ();
    return NULL;
  }

  /* Sleep until we reach time T'=T+period.  TODO: warn if current time
     is > T'.  */
  void
  wait_rest_of_period ()
  {
    m_timer.tv_nsec += m_period_ns;
    handle_timespec_overflow (&m_timer);

    clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &m_timer, NULL);
  }

  /* Local timer.  */
  timespec m_timer;
  /* To synchronise with everyone else at the start.  */
  pthread_barrier_t *m_barrier = nullptr;
  /* Killswitch.  */
  std::atomic<bool> m_alive {false};

  /* The compiler deletes the default CC when the class contains an atomic.  */
  rt_thread (const rt_thread& other)
    : m_timer (other.m_timer), m_barrier (other.m_barrier),
      m_alive (other.m_alive.load ()), m_period_ns (other.m_period_ns),
      m_id (other.m_id), m_priority (other.m_priority)
  {}

  virtual ~rt_thread () = default;

private:
  /* Copy of network-wide period parameter.  */
  uint64_t m_period_ns;
  /* pthread API info.  */
  pthread_t m_id;
  int m_priority;
};

/* The natural interval [BEGIN, END) represents a contigious part of
   layer L's neurons that we refer to as a 'sublayer'.  It is the job
   of each network thread to run a set of these.  */
struct sublayer
{
  layer *l;
  uint32_t begin;
  uint32_t end;

  sublayer (layer *l, uint32_t begin, uint32_t end)
    : l (l), begin (begin), end (end)
  {}
};

/* A thread which computes part of the network (see 'sublayer').  */
class network_rtt : public rt_thread
{
public:

  network_rtt (uint64_t period_ns, std::vector<sublayer> slayers,
	       int priority=80)
    : rt_thread (period_ns, priority), m_sublayers (slayers)
  {}

  void
  run ()
  {
    while (m_alive.load (std::memory_order_relaxed))
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

/* A thread which takes input from callback M_CB and writes it to a buffer
   M_BUFFER read by the thread(s) handling the input layer of the network.

   The input callback may kill the network by writing to it's boolean
   argument.  */
class input_rtt : public rt_thread
{
public:

  input_rtt (uint64_t period_ns, std::vector<uint32_t> (*cb) (bool *),
	     spikebuffer *buff, int priority=80)
    : rt_thread (period_ns, priority), m_buffer (buff), m_cb (cb)
  {}

  void
  run ()
  {
    bool killed = false;
    while (!killed && m_alive.load (std::memory_order_relaxed))
      {
	m_buffer->write (m_cb (&killed));
	wait_rest_of_period ();
      }

    /* Propagate death.  */
    m_alive.store (false, std::memory_order_relaxed);
  }

private:
  /* Written to by us only, read by threads running the first layer.  */
  spikebuffer *m_buffer = nullptr;
  /* Callback.  */
  std::vector<uint32_t> (*m_cb) (bool *) = nullptr;
};

/* A thread which reads from the buffer M_BUFFER written to by the thread(s)
   handling the output layer of the network and passes it to callback M_CB.  */
class output_rtt : public rt_thread
{
public:

  output_rtt (uint64_t period_ns, void (*cb) (const std::vector<uint32_t> &),
	      spikebuffer *buff, int priority=80)
    : rt_thread (period_ns, priority), m_buffer (buff), m_cb (cb)
  {}

  void
  run ()
  {
    while (m_alive.load (std::memory_order_relaxed))
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

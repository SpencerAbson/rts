#include <sched.h>
#include <pthread.h>
#include "../include/util.h"
#include "../include/threads.hpp"

uint32_t thread::m_debug_id_counter = 0;

thread::thread (uint32_t period_us)
  : m_period_ns ((uint64_t)period_us * 1E3), m_debug_id (m_debug_id_counter)
{
  m_debug_id_counter++;
}

uint32_t
thread::debug_id () const
{
  return m_debug_id;
}

int
thread::start (signal *start, signal *exit)
{
  m_start_notification = start;
  m_exit_notification = exit;
#ifdef RTS_EN_RT_POLICY
  return create_rt_pthread ();
#else
  return pthread_create (&m_id, NULL, runner, (void *)this);
#endif
}

int
thread::join ()
{
  int ret = pthread_join (m_id, NULL);
  if (ret)
    debug_msg ("Failed to join pthread.\n", ret);

  return ret;
}

int
thread::kill_join ()
{
  m_alive.store (false, std::memory_order_release);
  return join ();
}

int
thread::write_perf_metrics (const std::string &path_latencies,
			    const std::string &path_wakeups) const
{
#ifdef RTS_EN_PROFILE_NETWORK
  int ret = weights_to_file (path_latencies, m_latencies);
  if (ret)
    debug_msg ("Failed to write performance metrics to file: {}.\n",
	       path_latencies);

  ret = weights_to_file (path_wakeups, m_wakeup_times);
  if (ret)
    debug_msg ("Failed to write performance metrics to file: {}.\n",
	       path_wakeups);

  return ret;
#else
  debug_msg ("Warn: metrics requested but profiling is disabled.\n");
  return 0;
#endif
}

void
thread::complete_period ()
{
#ifdef RTS_EN_PROFILE_NETWORK
  timespec end;
  clock_gettime (CLOCK_MONOTONIC, &end);
  /* Record the latency of this cycle.  */
  uint64_t latency = (end.tv_sec - m_timer.tv_sec) * 1E9
    + end.tv_nsec - m_timer.tv_nsec;

  m_latencies.push_back (latency);
#endif
  timespec now;

  /* We should sleep until...  */
  m_timer.tv_nsec += m_period_ns;
  handle_timespec_overflow (&m_timer);

  /* If this value is earlier than the current time, we have a
     timing violation.  */
  clock_gettime (CLOCK_MONOTONIC, &now);
  if (time_is_later (now, m_timer))
    {
      debug_msg ("Fatal timing violation\n");
      /* Notify the main thread of our exit.  */
      m_exit_notification->post ();
      pthread_exit (NULL);
    }

  /* Otherwise, sleep until the next cycle.  */
  clock_nanosleep (CLOCK_MONOTONIC, TIMER_ABSTIME, &m_timer, NULL);

#ifdef RTS_EN_PROFILE_NETWORK
  /* Record the wakeup time.  */
  clock_gettime (CLOCK_MONOTONIC, &now);
  m_wakeup_times.push_back (now.tv_sec * 1E9 + now.tv_nsec);
#endif
}

int
thread::create_rt_pthread ()
{
  pthread_attr_t attr;
  int ret = pthread_attr_init (&attr);
  if (ret)
    {
      debug_perror ("pthread_attr_init");
      debug_msg ("Failed to initialise pthread attrs.\n");
      return ret;
    }

  /* Set scheduler policy.  */
  ret = pthread_attr_setschedpolicy (&attr, SCHED_FIFO);
  if (ret)
    {
      debug_perror ("pthread_attr_setschedpolicy");
      debug_msg ("Failed to set pthread scheduler policy.\n");
      return ret;
    }

  /* Set scheduler priority.  */
  struct sched_param param;
  param.sched_priority = 80;
  ret = pthread_attr_setschedparam (&attr, &param);
  if (ret)
    {
      debug_perror ("pthread_attr_setschedparam");
      debug_msg ("Failed to set pthread scheduler priority.\n");
      return ret;
    }

  /* Ensure pthread_create uses our ATTR rather than those of the
     parent thread.  */
  ret = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
  if (ret)
    {
      debug_perror ("pthread_attr_setinheritsched");
      debug_msg ("Failed to set pthread scheduler attributes.\n");
      return ret;
    }
  /* Create the thread.  */
  return pthread_create (&m_id, &attr, runner, (void *)this);
}


/* sublayer impl.  */
sublayer::sublayer (layer *l, uint32_t begin, uint32_t end)
  : l (l), begin (begin), end (end)
{}

std::string
sublayer::str_descr (uint32_t level) const
{
  return std::format ("{}(sublayer:{} {} (range {} {})\n{})",
		      std::string (level, '\t') , l->debug_type (),
		      l->debug_id (), begin, end, l->str_buffers (level + 1));
}


/* network_thread impl.  */
network_thread::network_thread (uint32_t period_us,
				std::vector<sublayer> slayers)
  : thread (period_us), m_sublayers (slayers)
{}

void
network_thread::run ()
{
  while (m_alive.load (std::memory_order_acquire))
    {
      for (sublayer &slayer : m_sublayers)
	slayer.l->run (slayer.begin, slayer.end);

      complete_period ();
    }
}

std::string
network_thread::str_descr (uint32_t level) const
{
  std::string temp = "";
  for (const sublayer &slayer: m_sublayers)
    temp += "\n" + slayer.str_descr (level + 1);

  std::string space = std::string (level, '\t');
  return std::format ("{}(thread:NET {} [{}\n{}])", space, m_debug_id,
		      temp, space);
}


/* input_thread impl.  */
input_thread::input_thread (uint32_t period_us, callback_type cb,
			    spikebuffer *buff)
  : thread (period_us), m_buffer (buff), m_cb (cb)
{}

void
input_thread::run ()
{
  std::vector<uint32_t> *spikes_wr;

  bool killed = false;
  while (true)
    {
      if (!m_alive.load (std::memory_order_acquire))
	/* We've been killed by another thread.  */
	break;

      std::vector<uint32_t> temp = m_cb (&killed);
      if (killed)
	/* We've been killed locally by M_CB.  */
	{
	  /* Propagate death.  */
	  m_alive.store (false, std::memory_order_release);
	  break;
	}

      /* Continue as normal.  */
      spikes_wr = m_buffer->acquire_write ();
      spikes_wr->assign (temp.begin (), temp.end ());
      m_buffer->release_write ();

      complete_period ();
    }
}

std::string
input_thread::str_descr (uint32_t level) const
{
  return std::format ("{}(thread:INP {} (buff:WR {}))",
		      std::string (level, '\t'), m_debug_id,
		      m_buffer->debug_id ());
}

/* output_thread impl.  */
output_thread::output_thread (uint32_t period_us, callback_type cb,
			      spikebuffer *buff)
  : thread (period_us), m_buffer (buff), m_cb (cb)
{}

void
output_thread::run ()
{
  while (m_alive.load (std::memory_order_acquire))
    {
      m_cb (*m_buffer->acquire_read ());
      m_buffer->release_read ();

      complete_period ();
    }
}

std::string
output_thread::str_descr (uint32_t level) const
{
  return std::format ("{}(thread:OUP {} (buff:RD {}))",
		      std::string (level, '\t'), m_debug_id,
		      m_buffer->debug_id ());
}

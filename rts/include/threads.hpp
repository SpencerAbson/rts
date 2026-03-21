#ifndef THREADS_H_
#define THREADS_H_

#include <atomic>
#include <functional>
#include "signal.hpp"
#include "buffers.hpp"
#include "layers/layer.hpp"

/* pthread_t wrapper.  */
class thread
{
  static uint32_t m_debug_id_counter;
public:
  thread (uint32_t period_us);
  virtual ~thread () = default;

  /* RT cyclic task implementation.  */
  virtual void
  run () = 0;

  /* debug printing.  */
  virtual std::string
  str_descr (uint32_t level=0) const = 0;

  uint32_t
  debug_id () const;

  /* Initialise and run, returning 0 on success and a pthread error
     code otherwise.  */
  int
  start (signal *start_notification, signal *exit_notification);
  /* Wrapper on pthread_join.  */
  int
  join ();

  /* Set M_ALIVE to false and join, returning 0 on success and a
     pthread error code otherwise.  */
  int
  kill_join ();

  int
  write_perf_metrics (const std::string &path_latencies,
		      const std::string &path_wakeups) const;

  static void*
  runner (void *arg)
  {
    thread *worker = (thread *)arg;
    /* Set state to alive.  */
    worker->m_alive.store (true, std::memory_order_relaxed);
    /* Notifiy every other thread that we are ready.  */
    worker->m_start_notification->post ();

    /* Sleep until every other thread is ready.  */
    if (worker->m_start_notification->wait ())
      /* The start was cancelled...  */
      {
	/* Notify the main thread of our exit.  */
	worker->m_exit_notification->post ();
	pthread_exit (NULL);
      }

    /* Set the timer.  */
    clock_gettime (CLOCK_MONOTONIC, &worker->m_timer);
#ifdef EN_PROFILE_NETWORK
    /* Record the initial wakeup time.  */
    worker->m_wakeup_times.push_back (worker->m_timer.tv_sec * 1E9
				      + worker->m_timer.tv_nsec);
#endif

    /* Run!  */
    worker->run ();

    /* Notify the main thread of our exit.  */
    worker->m_exit_notification->post ();
    pthread_exit (NULL);
  }

  /* Sleep until we reach time T'=T+period.  TODO: warn if current time
     is > T'.  */
  void
  complete_period ();

  /* Local timer.  */
  timespec m_timer;
  /* To synchronise with everyone else at the start.  */
  signal *m_start_notification;
  /* To pass the exiting signal back to the main thread.  */
  signal *m_exit_notification;

  /* Killswitch.  */
  std::atomic<bool> m_alive {false};

  /* The compiler deletes the default CC when the class contains an atomic.  */
  thread (const thread& other)
  {
    m_timer     = other.m_timer;
    m_alive     = other.m_alive.load (std::memory_order_seq_cst);
    m_period_ns = other.m_period_ns;
    m_id        = other.m_id;
    m_start_notification = other.m_start_notification;
    m_exit_notification  = other.m_exit_notification;
#ifdef EN_PROFILE_NETWORK
    m_latencies = other.m_latencies;
    m_wakeup_times = other.m_wakeup_times;
#endif
  }

private:
  /* Copy of network-wide period parameter.  */
  uint64_t m_period_ns;
  /* pthread ID.  */
  pthread_t m_id;
#ifdef EN_PROFILE_NETWORK
  std::vector<uint64_t> m_latencies;
  std::vector<uint64_t> m_wakeup_times;
#endif
  /* Create a pthread under the SCHED_FIFO policy.  */
  int
  create_rt_pthread ();

protected:
  uint32_t m_debug_id;
};


/* The natural interval [BEGIN, END) represents a contigious part of layer
   L's neurons that we refer to as a 'sublayer'.  It is the job of each
   network thread to run a set of these.  */
struct sublayer
{
  layer *l;
  uint32_t begin;
  uint32_t end;
  sublayer (layer *l, uint32_t begin, uint32_t end);

  /* debug printing.  */
  std::string str_descr (uint32_t level=0) const;
};

/* A thread which computes part of the network (see 'sublayer').  */
class network_thread : public thread
{
public:
  network_thread (uint32_t period_us, std::vector<sublayer> slayers);
  void
  run ();

  /* debug printing.  */
  std::string str_descr (uint32_t level=0) const;
private:
  /* Workload.  */
  std::vector<sublayer> m_sublayers;
};

/* A thread which takes input from callback M_CB and writes it to a buffer
   M_BUFFER read by the thread(s) handling the input layer of the network.

   The input callback may kill the network by writing to it's boolean
   argument.  */
class input_thread : public thread
{
public:
  using callback_type = std::function<std::vector<uint32_t> (bool *)>;

  input_thread (uint32_t period_us, callback_type cb, spikebuffer *buff);
  void
  run ();

  /* debug printing.  */
  std::string str_descr (uint32_t level=0) const;
private:
  /* Written to by us only, read by threads running the first layer.  */
  spikebuffer *m_buffer = nullptr;
  /* Callback.  */
  callback_type m_cb;
};

/* A thread which reads from the buffer M_BUFFER written to by the thread(s)
   handling the output layer of the network and passes it to callback M_CB.  */
class output_thread : public thread
{
public:
  using callback_type = std::function<void (const std::vector<uint32_t> &)>;

  output_thread (uint32_t period_us, callback_type cb, spikebuffer *buff);
  void
  run ();

  /* debug printing.  */
  std::string str_descr (uint32_t level=0) const;
private:
  /* Written to by threads running the last layer, read by us only.  */
  spikebuffer *m_buffer = nullptr;
  /* Callback.  */
  callback_type m_cb;
};

#endif // THREADS_H_

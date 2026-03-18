#ifndef THREADS_H_
#define THREADS_H_

#include <atomic>
#include <functional>
#include <semaphore.h>
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

  /* Initialise and run, returning 0 on success and a pthread error
     code otherwise.  */
  int
  start (pthread_barrier_t *barrier, sem_t *sema);
  /* Wrapper on pthread_join.  */
  int
  join ();
  /* Kill and join, returning 0 on success and a pthread error
     code otherwise.  */
  int
  kill ();

  static void*
  runner (void *arg)
  {
    thread *worker = (thread *)arg;
    rts_checking_assert (worker->m_barrier != nullptr);

    /* Set state to alive.  */
    worker->m_alive.store (true, std::memory_order_relaxed);
    /* Sleep until everyone else is ready.  */
    pthread_barrier_wait (worker->m_barrier);
    /* Set the timer.  */
    clock_gettime (CLOCK_MONOTONIC, &worker->m_timer);

    /* Run!  */
    worker->run ();

    /* Notify the main thread of our exit.  */
    sem_post (worker->m_notify_exit);
    pthread_exit (NULL);
  }

  /* Sleep until we reach time T'=T+period.  TODO: warn if current time
     is > T'.  */
  void
  complete_period ();

  /* Stringify the stats collected under EN_PROFILE_NETWORK.  */
  std::string
  str_perf_metrics () const;

  /* Local timer.  */
  timespec m_timer;
  /* To synchronise with everyone else at the start.  */
  pthread_barrier_t *m_barrier = nullptr;
  /* To pass the exiting signal back to the main thread.  */
  sem_t *m_notify_exit;

  /* Killswitch.  */
  std::atomic<bool> m_alive {false};

  /* The compiler deletes the default CC when the class contains an atomic.  */
  thread (const thread& other)
  {
    m_timer     = other.m_timer;
    m_barrier   = other.m_barrier;
    m_alive     = other.m_alive.load (std::memory_order_seq_cst);
    m_period_ns = other.m_period_ns;
    m_id        = other.m_id;
#ifdef EN_PROFILE_NETWORK
    m_max_latency_ns   = other.m_max_latency_ns;
    m_min_latency_ns   = other.m_min_latency_ns;
    m_total_latency_ns = other.m_total_latency_ns;
    m_total_cycles     = other.m_total_cycles;
#endif
  }

private:
  /* Copy of network-wide period parameter.  */
  uint64_t m_period_ns;
  /* pthread ID.  */
  pthread_t m_id;
  /* Create a pthread under the SCHED_FIFO policy.  */
  int
  create_rt_pthread ();
protected:
  uint32_t m_debug_id;
#ifdef EN_PROFILE_NETWORK
private:
  uint64_t m_max_latency_ns = 0;
  uint64_t m_min_latency_ns = UINT64_MAX;
  uint64_t m_total_latency_ns = 0;

  uint64_t m_total_cycles = 0;
#endif
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

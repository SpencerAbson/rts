#ifndef SIGNAL_H_
#define SIGNAL_H_

#include <stdint.h>
#include <pthread.h>

/* Waiting at a pthread_barrier_t is not cancellable.  This is a generalised
   broadcast/wait signalling data structure whose wait IS cancellable.  */
class signal
{
public:
  signal (uint32_t num_signallers);
  ~signal ();

  /* Increment M_NUM_SIGNALLED.  */
  void
  post ();
  /* Wait for M_NUM_SIGNALLED == M_NUM_SIGNALLERS.  Return -1
     if this wait was cancelled, and 0 on success.  */
  int
  wait ();

  /* Cancel any calls to wait ().  */
  void
  cancel ();

private:
  /* The number of times M_COND needs to be signalled before we stop
     waiting.  */
  uint32_t m_num_signallers;
  /* The number of times it has been signaled.  */
  uint32_t m_num_signalled = 0;
  /* Cancellation state.  */
  bool m_cancelled = false;

  pthread_mutex_t m_lock;
  pthread_cond_t  m_cond;
};

#endif // SIGNAL_H_

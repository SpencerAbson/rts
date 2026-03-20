#include "../include/util.h"
#include "../include/signal.hpp"

signal::signal (uint32_t num_signallers)
  : m_num_signallers (num_signallers)
{
  /* These always return 0.  */
  pthread_cond_init (&m_cond, NULL);
  pthread_mutex_init (&m_lock, NULL);
}

void
signal::post ()
{
  pthread_mutex_lock (&m_lock);

  m_num_signalled++;
  pthread_cond_broadcast (&m_cond);

  pthread_mutex_unlock (&m_lock);
}

int
signal::wait ()
{
  pthread_mutex_lock (&m_lock);

  while (m_num_signalled < m_num_signallers && !m_cancelled)
    pthread_cond_wait (&m_cond, &m_lock);

  int ret = m_cancelled ? -1 : 0;
  pthread_mutex_unlock (&m_lock);
  return ret;
}

void
signal::cancel ()
{
  pthread_mutex_lock (&m_lock);

  m_cancelled = true;
  pthread_cond_broadcast (&m_cond);

  pthread_mutex_unlock (&m_lock);
}

signal::~signal ()
{
  if (pthread_cond_destroy (&m_cond))
    debug_msg ("Failed to destroy condition variable");
  if (pthread_mutex_destroy (&m_lock))
    debug_msg ("Failed to destroy mutex");
}

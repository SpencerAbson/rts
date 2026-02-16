#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <vector>

/* A multi-producer multi-consumer double buffer to be set between successive
   layers in a network during simulation.  Built into this buffer is the logic
   required to detect meaningful timing violations.  */
class spikebuffer
{
public:

  spikebuffer (uint32_t writers, uint32_t readers, uint64_t sleep_ns=50000)
    : m_writers (writers), m_readers (readers)
  {
    m_sleep.tv_sec  = 0;
    m_sleep.tv_nsec = sleep_ns;
    handle_timespec_overflow (m_sleep);
  }

  spikebuffer (uint64_t sleep_ns=50000)
  {
    m_sleep.tv_sec  = 0;
    m_sleep.tv_nsec = sleep_ns;
    handle_timespec_overflow (m_sleep);
  }

  void
  set_readers (uint32_t readers)
  {
    m_readers = readers;
  }
  void
  set_writers (uint32_t writers)
  {
    m_writers = writers;
  }

  void
  write (const std::vector<uint32_t> &data_in)
  {
    /* Spin until we acquire the lock.  */
    while (std::atomic_exchange_explicit (&m_lock, true,
					  std::memory_order_acquire))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &m_sleep, NULL);

    if (m_tick)
      {
	/* There must have been a timing violation if the current write to
	   M_BUFF_A has already completed (we should be writing to M_BUFF_B
	   but never reached a stable state in the previous cycle).  */
	if (m_written_a == m_writers)
	  handle_timing_violation ();
	else
	  {
	    m_buff_a.insert (m_buff_a.end (), data_in.begin (), data_in.end ());
	    m_written_a++;
	  }
	/* Test for stability, and turnover if so.  */
	if (m_written_a == m_writers && m_read_b == m_readers)
	  tock ();
      }
    else
      {
	/* Similarly for M_BUFF_B.  */
	if (m_written_b == m_writers)
	  handle_timing_violation ();
	else
	  {
	    m_buff_b.insert (m_buff_b.end (), data_in.begin (), data_in.end ());
	    m_written_b++;
	  }

	if (m_written_b == m_writers && m_read_a == m_readers)
	  tick ();
      }
    /* Release.  */
    std::atomic_store_explicit (&m_lock, false, std::memory_order_release);
  }

  std::vector<uint32_t>
  read ()
  {
    std::vector<uint32_t> data_out;
    /* Spin until we acquire the lock.  */
    while (std::atomic_exchange_explicit (&m_lock, true,
					  std::memory_order_acquire))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &m_sleep, NULL);

    if (m_tick)
      {
	/* There must have been a timing violation if the current read from
	   M_BUFF_B has already completed (we should be reading from M_BUFF_A
	   but never reached a stable state in the previous cycle).  */
	if (m_read_b == m_readers)
	  handle_timing_violation ();
	else
	  {
	    data_out = m_buff_b;
	    m_read_b++;
	  }
	/* Test for stability, and turnover if so.  */
	if (m_written_a == m_writers && m_read_b == m_readers)
	  tock ();
      }
    else
      {
	/* Similarly for M_BUFF_A.  */
	if (m_read_a == m_readers)
	  handle_timing_violation ();
	else
	  {
	    data_out = m_buff_a;
	    m_read_a++;
	  }

	if (m_written_b == m_writers && m_read_a == m_readers)
	  tick ();
      }
    /* Release.  */
    std::atomic_store_explicit (&m_lock, false, std::memory_order_release);

    return data_out;
  }

private:

  void
  tick ()
  {
    /* Transition from writing to M_BUFF_B and reading from M_BUFF_A
       to reading from M_BUFF_B and writing to M_BUFF_A.  */
    m_read_b    = 0;
    m_written_a = 0;
    m_buff_a.clear ();
    m_tick = true;
  }

  void
  tock ()
  {
    /* Transition from writing to M_BUFF_A and reading from M_BUFF_B
       to reading from M_BUFF_A and writing to M_BUFF_B.  */
    m_read_a    = 0;
    m_written_b = 0;
    m_buff_b.clear ();
    m_tick = false;
  }

  void
  handle_timing_violation ()
  {
    exit (-1);
  }

  uint32_t m_writers = 0;
  uint32_t m_readers = 0;

  std::vector<uint32_t> m_buff_a;
  std::vector<uint32_t> m_buff_b;

  /* Read/Write state for M_BUFF_A.  */
  uint32_t m_read_a    = 0;
  uint32_t m_written_a = 0;
  /* Likewise for M_BUFF_B.  */
  uint32_t m_read_b    = 0;
  uint32_t m_written_b = 0;

  bool m_tick = false;
  timespec m_sleep;
  std::atomic<bool> m_lock = false;
};

#endif // BUFFERS_H_

#include "../include/util.h"
#include "../include/buffers.hpp"

uint32_t spikebuffer::m_debug_id_counter = 0;

spikebuffer::spikebuffer (uint32_t writers, uint32_t readers, tv_mechanism mech)
  : m_writers (writers), m_readers (readers), m_tv_mech (mech),
    m_debug_id (m_debug_id_counter)
{
  m_debug_id_counter++;
}

spikebuffer::spikebuffer (tv_mechanism mech)
  : m_tv_mech (mech), m_debug_id (m_debug_id_counter)
{
  m_debug_id_counter++;
}

void
spikebuffer::set_readers (uint32_t readers)
{
  m_readers = readers;
}

void
spikebuffer::set_writers (uint32_t writers)
{
  m_writers = writers;
}

void
spikebuffer::reserve (std::vector<uint32_t>::size_type size)
{
  m_buff_a.reserve (size);
  m_buff_b.reserve (size);
}

void
spikebuffer::write (const std::vector<uint32_t> &data_in)
{
  /* Spin until we acquire the lock.  */
  while (std::atomic_exchange_explicit (&m_lock, true,
					std::memory_order_acquire));
  if (m_tick)
    {
      /* There must have been a timing violation if the current write to
	 M_BUFF_A has already completed (we should be writing to M_BUFF_B
	 but never reached a stable state in the previous cycle).  */
      if (m_written_a == m_writers)
	{
	  if (!handle_timing_violation ())
	    {
	      /* Don't perform the write.  Release and return.  */
	      std::atomic_store_explicit (&m_lock, false,
					  std::memory_order_release);
	      return;
	    }
	}

      m_buff_a.insert (m_buff_a.end (), data_in.begin (), data_in.end ());
      m_written_a++;

      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers)
	tock ();
    }
  else
    {
      /* Similarly for M_BUFF_B.  */
      if (m_written_b == m_writers)
	{
	  if (!handle_timing_violation ())
	    {
	      /* Don't perform the write.  Release and return.  */
	      std::atomic_store_explicit (&m_lock, false,
					  std::memory_order_release);
	      return;
	    }
	}

      m_buff_b.insert (m_buff_b.end (), data_in.begin (), data_in.end ());
      m_written_b++;

      if (m_written_b == m_writers && m_read_a == m_readers)
	tick ();
    }
  /* Release.  */
  std::atomic_store_explicit (&m_lock, false, std::memory_order_release);
}

std::vector<uint32_t>
spikebuffer::read ()
{
  std::vector<uint32_t> data_out;
  /* Spin until we acquire the lock.  */
  while (std::atomic_exchange_explicit (&m_lock, true,
					std::memory_order_acquire));
  if (m_tick)
    {
      /* There must have been a timing violation if the current read from
	 M_BUFF_B has already completed (we should be reading from M_BUFF_A
	 but never reached a stable state in the previous cycle).  */
      if (m_read_b == m_readers)
	{
	  if (!handle_timing_violation ())
	    {
	      /* Don't perform the read.  Release and return.  */
	      std::atomic_store_explicit (&m_lock, false,
					  std::memory_order_release);
	      return data_out;
	    }
	}

      data_out = m_buff_b;
      m_read_b++;

      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers)
	tock ();
    }
  else
    {
      /* Similarly for M_BUFF_A.  */
      if (m_read_a == m_readers)
	{
	  if (!handle_timing_violation ())
	    {
	      /* Don't perform the read.  Release and return.  */
	      std::atomic_store_explicit (&m_lock, false,
					  std::memory_order_release);
	      return data_out;
	    }
	}

      data_out = m_buff_a;
      m_read_a++;

      if (m_written_b == m_writers && m_read_a == m_readers)
	tick ();
    }
  /* Release.  */
  std::atomic_store_explicit (&m_lock, false, std::memory_order_release);

  return data_out;
}

void
spikebuffer::wait_until_stable_acquire ()
{
  timespec sleep;
  sleep.tv_sec  = 0;
  /* FIXME: this value is arbitrary.  */
  sleep.tv_nsec = 200000;

  while (true)
    {
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
      /* Try to get the lock.  */
      while (std::atomic_exchange_explicit (&m_lock, true,
					    std::memory_order_acquire))
	continue;

      if (m_tick)
	{
	  if (m_written_a == m_writers && m_read_b == m_readers)
	    return;
	}
      else
	{
	  if (m_written_b == m_writers && m_read_a == m_readers)
	    return;
	}
      /* Release.  */
      std::atomic_store_explicit (&m_lock, false, std::memory_order_relaxed);
    }
}

void
spikebuffer::reset ()
{
  /* Reset the state to that after instantiation.  */
  tick ();
  tock ();
}

void
spikebuffer::tick ()
{
  /* Transition from writing to M_BUFF_B and reading from M_BUFF_A
     to reading from M_BUFF_B and writing to M_BUFF_A.  */
  m_read_b    = 0;
  m_written_a = 0;
  m_buff_a.clear ();
  m_tick = true;
}

void
spikebuffer::tock ()
{
  /* Transition from writing to M_BUFF_A and reading from M_BUFF_B
     to reading from M_BUFF_A and writing to M_BUFF_B.  */
  m_read_a    = 0;
  m_written_b = 0;
  m_buff_b.clear ();
  m_tick = false;
}

bool
spikebuffer::handle_timing_violation ()
{
  debug_msg ("Timing violation detected, buffer id:{}\n", m_debug_id);

  switch (m_tv_mech)
    {
    case tv_mechanism::WAIT:
      /* Release the lock and wait for stability.  */
      std::atomic_store_explicit (&m_lock, false, std::memory_order_release);
      wait_until_stable_acquire ();

      /* Continue.  */
      return true;
    case tv_mechanism::DROP:
      /* Don't continue.  */
      return false;
    case tv_mechanism::FATAL:
      rts_unreachable ("Fatal timing violation.");
    default:
      /* tv_mechanism::IGNORE.  */
      return true;
    }
}

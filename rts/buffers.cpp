#include "util.h"
#include "buffers.hpp"

uint32_t spikebuffer::m_debug_id_counter = 0;

spikebuffer::spikebuffer (uint32_t writers, uint32_t readers)
  : m_writers (writers), m_readers (readers), m_debug_id (m_debug_id_counter)
{
  m_debug_id_counter++;
}

spikebuffer::spikebuffer ()
  : m_debug_id (m_debug_id_counter)
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

void
spikebuffer::handle_timing_violation ()
{
  debug_msg ("Timing violation detected, instability condition:\n");
  if (m_tick)
    debug_msg ("readers: {}, read_b: {}, writers {}, written_a: {}\n",
	       m_readers, m_read_b, m_writers, m_written_a);
  else
    debug_msg ("readers: {}, read_a: {}, writers {}, written_b: {}\n",
	       m_readers, m_read_a, m_writers, m_written_b);
  /* What now?  Stabilise?  */
}

#include "../include/util.h"
#include "../include/buffers.hpp"

uint32_t spikebuffer::m_debug_id_counter = 0;

spikebuffer::spikebuffer (std::vector<uint32_t>::size_type size)
  : m_debug_id (m_debug_id_counter)
{
  if (pthread_spin_init (&m_lock, PTHREAD_PROCESS_PRIVATE))
    {
      debug_msg ("failed to initialise spinlock\n");
      exit (-1);
    }

  m_buff_a.reserve (size);
  m_buff_b.reserve (size);

  m_debug_id_counter++;
}

spikebuffer::spikebuffer (std::vector<uint32_t>::size_type size,
			  uint32_t readers, uint32_t writers)
  : m_writers (writers), m_readers (readers), m_debug_id (m_debug_id_counter)
{
  if (pthread_spin_init (&m_lock, PTHREAD_PROCESS_PRIVATE))
    {
      debug_msg ("failed to initialise spinlock\n");
      exit (-1);
    }

  m_buff_a.reserve (size);
  m_buff_b.reserve (size);

  m_debug_id_counter++;
}

spikebuffer::~spikebuffer ()
{
  pthread_spin_destroy (&m_lock);
}

uint32_t
spikebuffer::debug_id () const
{
  return m_debug_id;
}

uint32_t
spikebuffer::readers () const
{
  return m_readers;
}

uint32_t
spikebuffer::writers () const
{
  return m_writers;
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

const std::vector<uint32_t> *
spikebuffer::acquire_read ()
{
  const std::vector<uint32_t> *out;

  pthread_spin_lock (&m_lock);

  if (m_tick)
    out = &m_buff_b;
  else
    out = &m_buff_a;

  m_active_rd++;
  pthread_spin_unlock (&m_lock);
  return out;
}

void
spikebuffer::release_read ()
{
  pthread_spin_lock (&m_lock);

  rts_checking_assert (m_active_rd != 0);
  m_active_rd--;

  if (m_tick)
    {
      m_read_b++;
      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers
	  && !m_active_rd)
	tock ();
    }
  else
    {
      m_read_a++;
      /* Likewise.  */
      if (m_written_b == m_writers && m_read_a == m_readers
	  && !m_active_rd)
	tick ();
    }

  pthread_spin_unlock (&m_lock);
}

std::vector<uint32_t> *
spikebuffer::acquire_write ()
{
  pthread_spin_lock (&m_lock);

  if (m_tick)
      return &m_buff_a;
  else
      return &m_buff_b;
}

void
spikebuffer::release_write ()
{
  if (m_tick)
    {
      m_written_a++;
      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers
	  && !m_active_rd)
	tock ();
    }
  else
    {
      m_written_b++;
      /* Likewise.  */
      if (m_written_b == m_writers && m_read_a == m_readers
	  && !m_active_rd)
	tick ();
    }

  pthread_spin_unlock (&m_lock);
}

void
spikebuffer::reset ()
{
  /* It's too fun not to write it like this.  */
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

#include "../include/util.h"
#include "../include/buffers.hpp"

uint32_t spikebuffer::m_debug_id_counter = 0;

spikebuffer::spikebuffer (std::vector<uint32_t>::size_type size,
			  tv_mechanism tv_mech)
  : m_tv_mech (tv_mech), m_debug_id (m_debug_id_counter)
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
			  uint32_t readers, uint32_t writers,
			  tv_mechanism tv_mech)
  : m_writers (writers), m_readers (readers), m_tv_mech (tv_mech),
    m_debug_id (m_debug_id_counter)
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
    {
      if (m_read_b == m_readers && !handle_read_tv ())
	/* There must have been a timing violation if the current read from
	   M_BUFF_B has already completed (we should be reading from M_BUFF_A
	   but never reached a stable state in the previous cycle).  */
	{
	  pthread_spin_unlock (&m_lock);
	  return nullptr;
	}

      out = &m_buff_b;
    }
  else
    {
      if (m_read_a == m_readers && !handle_read_tv ())
	{
	  pthread_spin_unlock (&m_lock);
	  return nullptr;
	}

      out = &m_buff_a;
    }

  pthread_spin_unlock (&m_lock);
  return out;
}

void
spikebuffer::release_read ()
{
  pthread_spin_lock (&m_lock);

  if (m_tick)
    {
      m_read_b++;
      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers)
	tock ();
    }
  else
    {
      m_read_a++;
      /* Test for stability, and turnover if so.  */
      if (m_written_b == m_writers && m_read_a == m_readers)
	tick ();
    }

  pthread_spin_unlock (&m_lock);
}

bool
spikebuffer::handle_read_tv ()
{
  debug_msg ("Timing violation detected on read, buffer id: {}\n",
	     m_debug_id);

  switch (m_tv_mech)
    {
    case tv_mechanism::DROP:
      /* Do not continue.  */
      return false;
    case tv_mechanism::IGNORE:
      /* Continue as normal.  */
      return true;
    default:
      debug_msg ("Fatal timing violation\n");
      exit (-1);
    }
}

std::vector<uint32_t> *
spikebuffer::acquire_write ()
{
  pthread_spin_lock (&m_lock);

  if (m_tick)
    {
      /* There must have been a timing violation if the current write to
	 M_BUFF_A has already completed (we should be writing to M_BUFF_B
	 but never reached a stable state in the previous cycle).  */
      if (m_written_a == m_writers)
	return handle_write_tv ();
      else
	return &m_buff_a;
    }
  else
    {
      /* Similarly for M_BUFF_B.  */
      if (m_written_b == m_writers)
	return handle_write_tv ();
      else
	return &m_buff_b;
    }
}

void
spikebuffer::release_write ()
{
  if (m_tick)
    {
      m_written_a++;
      /* Test for stability, and turnover if so.  */
      if (m_written_a == m_writers && m_read_b == m_readers)
	tock ();
    }
  else
    {
      m_written_b++;
      /* Likewise.  */
      if (m_written_b == m_writers && m_read_a == m_readers)
	tick ();
    }

  pthread_spin_unlock (&m_lock);
}

std::vector<uint32_t> *
spikebuffer::handle_write_tv ()
{
  debug_msg ("Timing violation detected on write, buffer id: {}\n",
	     m_debug_id);

  switch (m_tv_mech)
    {
    case tv_mechanism::DROP:
      return nullptr;
    case tv_mechanism::IGNORE:
      return m_tick ? &m_buff_a : &m_buff_b;
    default:
      debug_msg ("Fatal timing violation\n");
      exit (-1);
    }
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

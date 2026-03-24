#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <vector>
#include <pthread.h>

/* A multi-producer multi-consumer double buffer to be set between
   successive layers in a network during simulation.

   The idea is that this double-buffer manages its own swapping without
   interference from the network.  That is, it switches the active read
   and write buffers over at the end/start of every cycle by itself.

   It does this by checking for 'stability' after any read or write has
   completed.  The spikebuffer is stable when every reading thread has
   read the contents of the active read buffer, and every writing thread
   has written to the contents of the active write buffer.

   If stability is never reached within a cycle, then we have a timing-
   violation.  This will be detected by the network as soon as the
   violating thread completes its work.  We have an additional flag,
   M_ACTIVE_RD, to ensure that an illegal state during reached in the
   interval between the timing violation and simulation death does not
   result in a data race.  */

class spikebuffer
{
  static uint32_t m_debug_id_counter;
public:
  spikebuffer (std::vector<uint32_t>::size_type size);

  spikebuffer (std::vector<uint32_t>::size_type size, uint32_t readers,
	       uint32_t writers);

  ~spikebuffer ();

  uint32_t
  debug_id () const;

  uint32_t
  readers () const;
  uint32_t
  writers () const;

  void
  set_readers (uint32_t readers);
  void
  set_writers (uint32_t writers);

  /* READ ACCESS
     ...

     // Put the address of the shared active read buffer in DATA, unlocking
     // the spinlock again before returning.
     std::vector<uint32_t> *data = buffer->acquire_read ()

     ...

     // Relinquish read access to DATA, record the read as completed and
     // test for stability, unlocking the spinlock again before returning.
     buffer_release_readd ();  */
  const std::vector<uint32_t> *
  acquire_read ();
  void
  release_read ();

  /* WRITE ACCESS
     ...

     // Put the address of the shared active read buffer in DATA, leaving
     // the spinlock LOCKED.
     std::vector<uint32_t> *data = buffer->acquire_write ()

     ...

     // Relinquish write access to DATA, record the read as completed,
     // test for stability, and UNLOCK the spinlock.
     buffer->release_write ();  */
  std::vector<uint32_t> *
  acquire_write ();
  void
  release_write ();

  void
  reset ();

private:
  void
  tick ();
  void
  tock ();

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

  uint32_t m_active_rd = 0;

  bool m_tick = false;

  pthread_spinlock_t m_lock;
  uint32_t m_debug_id;
};


#endif // BUFFERS_H_

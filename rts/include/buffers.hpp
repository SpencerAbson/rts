#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <vector>
#include <pthread.h>


/* A multi-producer multi-consumer double buffer to be set between successive
   layers in a network during simulation.  */
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

  const std::vector<uint32_t> *
  acquire_read ();
  void
  release_read ();

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

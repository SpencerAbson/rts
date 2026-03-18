#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <vector>
#include <pthread.h>

/* What to do in the case of a timing violation.  */
enum tv_mechanism
{
  IGNORE, /* Nothing.  */
  DROP,   /* Drop the current read/write.  */
  FATAL   /* Exit the process.  */
};

/* A multi-producer multi-consumer double buffer to be set between successive
   layers in a network during simulation.  Built into this buffer is the logic
   required to detect meaningful timing violations.  */
class spikebuffer
{
  static uint32_t m_debug_id_counter;
public:
  spikebuffer (std::vector<uint32_t>::size_type size,
	       tv_mechanism tv_mech=DROP);

  spikebuffer (std::vector<uint32_t>::size_type size, uint32_t readers,
	       uint32_t writers, tv_mechanism tv_mech=DROP);

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

  bool
  handle_read_tv ();
  std::vector<uint32_t> *handle_write_tv ();

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

  tv_mechanism m_tv_mech;

  pthread_spinlock_t m_lock;
  uint32_t m_debug_id;
};


#endif // BUFFERS_H_

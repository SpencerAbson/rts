#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <vector>

/* What to do in the case of a timing violation.  */
enum tv_mechanism
{
  IGNORE, /* Nothing.  */
  DROP,   /* Drop the current read/write.  */
  WAIT,   /* Wait for stability.  */
  FATAL   /* Exit the process.  */
};

/* A multi-producer multi-consumer double buffer to be set between successive
   layers in a network during simulation.  Built into this buffer is the logic
   required to detect meaningful timing violations.  */
class spikebuffer
{
  static uint32_t m_debug_id_counter;
public:
  spikebuffer (uint32_t writers, uint32_t readers, tv_mechanism mech=DROP);
  spikebuffer (tv_mechanism mech=DROP);

  void
  set_readers (uint32_t readers) { m_readers = readers; }
  void
  set_writers (uint32_t writers) { m_writers = writers; }

  uint32_t
  readers () const { return m_readers; }
  uint32_t
  writers () const { return m_writers; }

  /* To avoid dynamic memory allocation on the RT critical path.  */
  void
  reserve (std::vector<uint32_t>::size_type size);

  void
  write (const std::vector<uint32_t> &data_in);
  std::vector<uint32_t>
  read ();

  /* Reset the state to that after instantiation.  */
  void
  reset ();

  uint32_t
  debug_id () const
  {
    return m_debug_id;
  }

private:
  void
  tick ();
  void
  tock ();

  /* Sleep wait until a stable state is reached.  Return without releasing
     the lock.  */
  void
  wait_until_stable_acquire ();

  /* Handle the timing violation w.r.t M_TV_MECH.  Return true if we
     should continue with the operation, or false otherwise.  */
  bool
  handle_timing_violation ();

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
  std::atomic<bool> m_lock {false};

  tv_mechanism m_tv_mech;
  uint32_t m_debug_id;
};

#endif // BUFFERS_H_

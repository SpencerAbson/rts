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

  spikebuffer (uint32_t writers, uint32_t readers, uint64_t sleep_ns=5000);
  spikebuffer (uint64_t sleep_ns=5000);

  void
  set_readers (uint32_t readers);
  void
  set_writers (uint32_t writers);
  /* To avoid dynamic memory allocation on the RT critical path.  */
  void
  reserve (std::vector<uint32_t>::size_type size);

  void
  write (const std::vector<uint32_t> &data_in);
  std::vector<uint32_t>
  read ();

private:
  void
  tick ();
  void
  tock ();
  void
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
  timespec m_sleep;
  std::atomic<bool> m_lock {false};
};

#endif // BUFFERS_H_

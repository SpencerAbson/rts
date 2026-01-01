#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <memory>
#include <vector>

template<typename T>
class buffer
{
public:

  virtual bool write (const std::vector<T>&) = 0;
  virtual bool latch (std::vector<T>&) = 0;
  virtual ~buffer () = default;
};


/* Single-producer single-consumer buffer.  */
template<typename T>
class spsc_buffer : public buffer<T>
{
public:

  bool
  write (const std::vector<T> &data_in)
  {
    if (!m_valid.load (std::memory_order_acquire))
      {
	m_buff.assign (data_in.begin (), data_in.end ());
	m_valid.store (true, std::memory_order_release);
	return true;
      }

    return false;
  }

  bool
  latch (std::vector<T> &data_out)
  {
    if (m_valid.load (std::memory_order_acquire))
      {
	data_out.assign (m_buff.begin (), m_buff.end ());
	m_valid.store (false, std::memory_order_release);
	return true;
      }

    return false;
  }

private:

  std::vector<T> m_buff;
  std::atomic<bool> m_valid;
};

#endif // BUFFERS_H_

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
};


template<typename T>
class spsc_buffer : public buffer<T>
{
public:

  bool
  write (const std::vector<T> &data_in)
  {
    /* If it's been latched, we're free to write.  */
    if (!valid.load (std::memory_order_acquire))
      {
	buff.assign (data_in.begin (), data_in.end ());
	valid.store (true, std::memory_order_release);
	return true;
      }

    return false;
  }

  bool
  latch (std::vector<T> &data_out)
  {
    if (valid.load (std::memory_order_acquire))
      {
	data_out.assign (buff.begin (), buff.end ());
	valid.store (false, std::memory_order_release);
	return true;
      }

    return false;
  }

private:

  std::vector<T> buff;
  std::atomic<bool> valid;
};

#endif // BUFFERS_H_

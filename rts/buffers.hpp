#ifndef BUFFERS_H_
#define BUFFERS_H_

#include <atomic>
#include <memory>
#include <vector>

template<typename T>
class buffer
{
public:

  virtual void write (const std::vector<T>&) = 0;
  virtual bool latch (std::vector<T>&) = 0;
};


/* Single producer (and comsumer) buffer.  */
template<typename T>
class sp_buffer : public buffer<T>
{
public:

  void
  write (const std::vector<T> &data_in)
  {
    /* If it's been latched, we're free to write.  */
    if (!valid.load (std::memory_order_acquire))
      {
	buff.assign (data_in.begin (), data_in.end ());
	valid.store (true, std::memory_order_release);
      }
    else
      assert (false);   /* Not sufficient slack?  */
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

    /* Less severe, perhaps no input fed into the net.  */
    return false;
  }

private:

  std::vector<T> buff;
  std::atomic<bool> valid;
};

#endif // BUFFERS_H_

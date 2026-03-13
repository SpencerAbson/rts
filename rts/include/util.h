#ifndef UTIL_H_
#define UTIL_H_

#include <format>
#include <memory>
#include <fstream>
#include <iostream>
#include <cassert>
#include <vector>
#include <time.h>
#include <stdio.h>

#define rts_unreachable(msg)			\
  assert (false && "Unreachable executed: " msg)

#ifdef EN_CHECKING_ASSERT
#define rts_checking_assert(EXPR)		\
    assert (EXPR)
#else
#define rts_checking_assert(EXPR)		\
    ((void)(0 && EXPR))
#endif

#ifdef EN_DEBUG_PRINT
#define debug_msg(fmt, ...)			 \
  std::cerr << std::format ("{}:{}:{} (): " fmt, \
    __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define debug_dump(fmt, ...)			\
  std::cerr << std::format (fmt, __VA_ARGS__)
#define debug_perror(msg)			\
  perror (msg)
#else
#define debug_msg(fmt, ...)
#define debug_dump(fmt, ...)
#define debug_perror(msg)
#endif


inline void
handle_timespec_overflow (timespec *ts)
{
  while (ts->tv_nsec >= 1000000000)
  {
    ts->tv_sec++;
    ts->tv_nsec -= 1000000000;
  }
}

template<typename T>
inline int
weights_from_file (std::string path, size_t count, std::vector<T> &out)
{
  std::ifstream file (path, std::ifstream::binary);
  if (!file.is_open ())
    return -1;

  out.resize (count);
  file.read ((char *)out.data (), count * sizeof(T));

  file.close ();
  return 0;
}

#endif // UTIL_H_

#ifndef UTIL_H_
#define UTIL_H_

#include <format>
#include <memory>
#include <random>
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
weights_from_file (std::string path, std::streamsize count,
		   std::vector<T> &out)
{
  std::ifstream file (path, std::ifstream::binary);
  if (!file.is_open ())
    return -1;

  int ret = 0;
  out.resize (count);
  if (!file.read ((char *)out.data (), count * sizeof (T)))
    ret = -1;

  file.close ();
  return ret;
}

inline std::mt19937&
mersenne_twister ()
{
  static std::random_device rd;
  static std::mt19937 g (rd ());
  return g;
}

#endif // UTIL_H_

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <cassert>

#ifdef EN_CHECKING_ASSERT
#define rts_checking_assert(EXPR)		\
    assert (EXPR)
#else
#define rts_checking_assert(EXPR)		\
    ((void)(0 && EXPR))
#endif

#ifdef EN_DEBUG_PRINT
#define debug_printf(fmt, ...)			\
  fprintf (stderr, "%s:%d:%s (): " fmt,		\
    __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define debug_perror(msg)			\
  perror (msg)
#else
#define debug_printf(fmt, ...)
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

#endif // UTIL_H_

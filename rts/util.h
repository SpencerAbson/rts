#ifndef UTIL_H_
#define UTIL_H_

#include <cassert>
#include <iostream>

#ifdef EN_CHECKING_ASSERT
#define rts_checking_assert(EXPR)		\
    assert (EXPR)
#else
#define rts_checking_assert(EXPR)		\
    ((void)(0 && EXPR))
#endif

inline void*
rts_malloc (size_t size)
{
  void *ret = malloc (size);
  if (!ret)
    {
      std::cerr << "rts: (malloc) failed to allocate " << size
		<< " bytes.\n";
      abort ();
    }

  return ret;
}

inline void
handle_timespec_overflow (timespec &ts)
{
  while (ts.tv_nsec >= 1000000000)
  {
    ts.tv_sec++;
    ts.tv_nsec -= 1000000000;
  }
}

#endif // UTIL_H_

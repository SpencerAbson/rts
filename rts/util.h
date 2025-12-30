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

#define ABSDIFF(a, b)		\
    (a > b) ? (a - b) : (b - a)

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

#endif // UTIL_H_

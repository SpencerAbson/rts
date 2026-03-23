#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <vector>
#include <cassert>
#include <algorithm>

using size_type = std::vector<uint32_t>::size_type;

std::vector<size_type>
maximal_indices (const std::vector<uint32_t> &vec)
{
  if (vec.empty ())
    return {};

  std::vector<size_type> indices;
  uint32_t max_val = *std::max_element (vec.begin (), vec.end ());

  for (size_type i = 0; i < vec.size (); ++i)
    {
      if (vec[i] == max_val)
	indices.push_back (i);
    }

  return indices;
}

template<typename T>
inline bool
vector_same_contents_p (const std::vector<T> &a,
			const std::vector<T> &b)
{
  if (a.size () != b.size ())
    return false;

  return std::is_permutation (a.begin (), a.end (), b.begin ());
}

double
kendall_rank_corr (const std::vector<uint32_t> &a,
		   const std::vector<uint32_t> &b)
{
  assert (!a.empty () && a.size () == b.size ());

  uint32_t concordant = 0;
  uint32_t discordant = 0;
  for (size_type i = 0; i < a.size (); i++)
    {
      for (size_type j = i + 1; j < a.size (); j++)
	{
	  if ((a[i] <= a[j]) == (b[i] <= b[j]))
	    concordant++;
	  else
	    discordant++;
	}
    }

  return (double)(concordant - discordant)
    / (a.size () * (a.size () - 1) / 2);
}

#endif // TEST_UTIL_H_

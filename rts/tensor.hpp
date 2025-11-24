#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdint.h>
#include <vector>
#include <numeric>
#include "util.h"

/* Generic tensor wrapper.  */
template <typename T>
class tensor
{
public:

  tensor (std::vector<T> data, std::vector<uint32_t> shape)
    : vec (data),
      shape (shape),
      stride (shape.size ())
  {
    rts_checking_assert (std::reduce (shape.begin (), shape.end (), 1,
				      std::multiplies ())
			 == vec.size ());
    init_stride (0);
  }

  tensor (std::vector<T> data, std::initializer_list<uint32_t> shape)
    : vec (data),
      shape (shape),
      stride (shape.size ())
  {
    rts_checking_assert (std::reduce (shape.begin (),shape.end (), 1,
				      std::multiplies ())
			 == vec.size ());
    init_stride (0);
  }

  tensor (std::vector<uint32_t> shape, T init=T{})
    : vec (std::reduce (shape.begin (), shape.end (), 1,
			std::multiplies ()), init),
      shape (shape),
      stride (shape.size ())
  {
    init_stride (0);
  }

  tensor (std::initializer_list<uint32_t> shape, T init=T{})
    : vec (std::reduce (shape.begin (), shape.end (), 1,
			std::multiplies ()), init),
      shape (shape),
      stride (shape.size ())
  {
    init_stride (0);
  }

  tensor () = default;

  std::vector<T> vec;
  std::vector<uint32_t> shape;
  std::vector<uint32_t> stride;

private:

  uint32_t
  init_stride (uint32_t i)
  {
    rts_checking_assert (i < shape.size ());

    uint32_t stride_i;
    if (i == (shape.size () - 1))
      stride_i = 1;
    else
      stride_i = init_stride (i + 1) * shape[i + 1];

    stride[i] = stride_i;
    return stride_i;
  }
};

#endif // TENSOR_H_

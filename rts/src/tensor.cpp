#include <numeric>
#include <arm_neon.h>
#include "../include/tensor.hpp"

template<typename T>
tensor<T>::tensor (std::vector<T> data, std::vector<uint32_t> shape)
  : vec (data), shape (shape), stride (shape.size ())
{
  rts_checking_assert (std::reduce (shape.begin (), shape.end (), 1,
				    std::multiplies ()) == vec.size ());
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::vector<T> data, std::initializer_list<uint32_t> shape)
  : vec (data), shape (shape), stride (shape.size ())
{
  rts_checking_assert (std::reduce (shape.begin (),shape.end (), 1,
				    std::multiplies ()) == vec.size ());
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::vector<uint32_t> shape, T init)
  : vec (std::reduce (shape.begin (), shape.end (), 1,
		      std::multiplies ()), init), shape (shape),
    stride (shape.size ())
{
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::initializer_list<uint32_t> shape, T init)
  : vec (std::reduce (shape.begin (), shape.end (), 1, std::multiplies ()),
	 init), shape (shape), stride (shape.size ())
{
  init_stride (0);
}

template<typename T>
uint32_t
tensor<T>::init_stride (uint32_t i)
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

template class tensor<uint32_t>;
template class tensor<float32_t>;
template class tensor<float16_t>;

#include <numeric>
#include <arm_neon.h>
#include "../include/util.h"
#include "../include/tensor.hpp"

#define SHAPE_SIZE(shape)						\
  std::reduce (shape.begin (),shape.end (), tensor<T>::size_type{1},	\
	       std::multiplies ())


template<typename T>
tensor<T>::tensor (std::string path, std::vector<tensor<T>::size_type> shape)
  : shape (shape), stride (shape.size ())
{
  int ret = weights_from_file (path, SHAPE_SIZE (shape), vec);
  assert (!ret && "Failed to load weights from file.");

  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::vector<T> data, std::vector<tensor<T>::size_type> shape)
  : vec (data), shape (shape), stride (shape.size ())
{
  rts_checking_assert (SHAPE_SIZE (shape) == vec.size ());
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::vector<T> data,
		   std::initializer_list<tensor<T>::size_type> shape)
  : vec (data), shape (shape), stride (shape.size ())
{
  rts_checking_assert (SHAPE_SIZE (shape) == vec.size ());
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::vector<tensor<T>::size_type> shape, T init)
  : vec (SHAPE_SIZE (shape), init), shape (shape),
    stride (shape.size ())
{
  init_stride (0);
}

template<typename T>
tensor<T>::tensor (std::initializer_list<tensor<T>::size_type> shape, T init)
  : vec (SHAPE_SIZE (shape), init), shape (shape), stride (shape.size ())
{
  init_stride (0);
}

template<typename T>
void
tensor<T>::reshape (std::initializer_list<tensor<T>::size_type> new_shape)
{
  rts_checking_assert (SHAPE_SIZE (new_shape) == vec.size ());

  shape = new_shape;
  stride.resize (shape.size ());
  init_stride (0);
}

template<typename T>
tensor<T>::size_type
tensor<T>::init_stride (tensor<T>::size_type i)
{
  rts_checking_assert (i < shape.size ());

  tensor<T>::size_type stride_i;
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

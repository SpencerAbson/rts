#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdint.h>
#include <vector>
#include "util.h"

/* Generic tensor wrapper.  */
template <typename T>
class tensor
{
public:

  tensor (std::vector<T> data, std::vector<uint32_t> shape);
  tensor (std::vector<T> data, std::initializer_list<uint32_t> shape);
  tensor (std::vector<uint32_t> shape, T init=T{});
  tensor (std::initializer_list<uint32_t> shape, T init=T{});
  tensor () = default;

  std::vector<T> vec;
  std::vector<uint32_t> shape;
  std::vector<uint32_t> stride;

private:

  uint32_t
  init_stride (uint32_t i);
};

#endif // TENSOR_H_

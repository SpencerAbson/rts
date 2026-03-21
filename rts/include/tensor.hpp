#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "util.h"


/* Generic tensor wrapper.  */
template <typename T>
class tensor
{
public:
  using size_type = std::vector<T>::size_type;

  tensor (const std::string &path, std::initializer_list<size_type> shape);
  tensor (std::vector<T> data, std::vector<size_type> shape);
  tensor (std::vector<T> data, std::initializer_list<size_type> shape);
  tensor (std::vector<size_type> shape, T init=T{});
  tensor (std::initializer_list<size_type> shape, T init=T{});
  tensor () = default;

  void
  reshape (std::initializer_list<size_type> shape);

  std::vector<T> vec;
  std::vector<size_type> shape;
  std::vector<size_type> stride;
private:

  size_type
  init_stride (size_type i);
};

#endif // TENSOR_H_

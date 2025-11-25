#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include "../buffers.hpp"

class layer
{
public:

  virtual uint32_t input_size () const = 0;
  virtual uint32_t output_size () const = 0;

  /* Some metric to estimate how much work is required.  */
  virtual uint32_t timestep_cost () const = 0;

  /* Execute a subbatch of this layer.  */
  virtual void timestep (std::unique_ptr<buffer<uint32_t>>,
			 std::unique_ptr<buffer<uint32_t>>,
			 uint32_t, uint32_t) = 0;
};

#endif // LAYER_H_

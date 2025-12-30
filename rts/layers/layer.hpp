#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include "../buffers.hpp"

#define UNIT_COST 1

class layer
{
public:

  virtual uint32_t input_size () const = 0;
  virtual uint32_t output_size () const = 0;

  /* A profile based estimate of the cost of running TIMESTEP across the
     entire layer, or UNIT_COST if we lack this information.  */
  virtual uint32_t timestep_cost () const
  {
    return UNIT_COST;
  }
  /* Simulate one timestep of the entire layer.  */
  virtual std::vector<uint32_t> timestep (const std::vector<uint32_t>&) = 0;
  /* Simulate one timestep for a subbatch of this layer.  */
  virtual std::vector<uint32_t> timestep_batched (const std::vector<uint32_t>&,
						  uint32_t, uint32_t) = 0;
};

#endif // LAYER_H_

#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include "../buffers.hpp"

#define COST_UNDEF 0


class layer
{
public:
  layer (uint64_t ts_cost_ns)
    : m_ts_cost_ns (ts_cost_ns)
  {}

  uint64_t timestep_cost () const
  {
    return m_ts_cost_ns;
  }

  virtual uint32_t input_size () const = 0;
  virtual uint32_t output_size () const = 0;

  /* A profile based estimate of the cost of running TIMESTEP across the
     entire layer, or COST_UNDEF if we lack this information.  */
  /* Simulate one timestep of the entire layer.  */
  virtual std::vector<uint32_t> timestep (const std::vector<uint32_t>&) = 0;
  /* Simulate one timestep for a subbatch of this layer.  */
  virtual std::vector<uint32_t> timestep_batched (const std::vector<uint32_t>&,
						  uint32_t, uint32_t) = 0;

  /* Many factors influence performance at runtime.  But in the absence of user-
     provided profile information (COST_UNDEF), a rough estimate can be made by
     profiling each layer at their own worst case input.  */
  virtual void time_worstcase (uint32_t iterations, struct timespec &tstart,
			      struct timespec &tend) = 0;
  void
  profile (uint32_t iterations)
  {
    struct timespec start, end;
    time_worstcase (iterations, start, end);

    m_ts_cost_ns = ((end.tv_sec - start.tv_sec) * 1000000000
      + end.tv_nsec - start.tv_nsec) / iterations;
  }

  virtual ~layer () = default;
private:
  uint64_t m_ts_cost_ns;
};

#endif // LAYER_H_

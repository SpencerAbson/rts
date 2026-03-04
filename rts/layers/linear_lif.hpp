#ifndef LINEAR_LIF_H_
#define LINEAR_LIF_H_

#include "layer.hpp"


template <typename T>
class linear_lif : public layer
{
public:
  linear_lif (tensor<T> weights, std::vector<T> bias, uint32_t batch_size,
	      T beta=(T)0.8, T v_thresh=(T)1.0, uint64_t batch_cost=COST_UNDEF);

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin, uint32_t batch_end);

  uint64_t
  time_batch_worstcase_ns ();

private:

  /* Linear parameters.  */
  tensor<T> m_weights;
  std::vector<T> m_bias;

  /* LIF dynamics.  */
  T m_beta;
  T m_v_thresh;
  std::vector<T> m_v_membrane;
};

#endif // LINEAR_LIF_H_

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
		    uint32_t batch_begin);

  std::vector<uint32_t>
  worstcase_input ();

  void
  reset ();

  std::vector<T> m_v_membrane;

private:
  /* Update the LIF update rule to each neuron.  */
  void
  f32_neuron_update (uint32_t batch_begin);
  void
  f16_neuron_update (uint32_t batch_begin);

  /* Apply the weighted contribution of SPIKES_IN.  */
  void
  f32_spike_prop (const std::vector<uint32_t> &spikes_in,
		  uint32_t batch_begin);
  void
  f16_spike_prop (const std::vector<uint32_t> &spikes_in,
		  uint32_t batch_begin);

  /* Linear parameters.  */
  tensor<T> m_weights;
  std::vector<T> m_bias;

  /* LIF dynamics.  */
  T m_beta;
  T m_v_thresh;
};

#endif // LINEAR_LIF_H_

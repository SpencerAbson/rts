#ifndef R_LINEAR_LIF_H_
#define R_LINEAR_LIF_H_

#include <arm_neon.h>
#include "layer.hpp"

/* A fully-connected layer of LIF neurons, each with a one-to-one recurrent
   connection.  */
template <typename T>
class r_linear_lif : public layer
{
public:
  r_linear_lif (tensor<T> weights, std::vector<T> bias, std::vector<T> w_rec,
		uint32_t batch_size, T beta=(T)0.8, T v_thresh=(T)1.0,
		uint64_t batch_cost=COST_UNDEF);

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin);

  std::vector<uint32_t>
  worstcase_input ();

  void
  reset ();

private:
  /* Linear parameters.  */
  tensor<T> m_weights;
  std::vector<T> m_bias;

  /* Recurrent parameters.  */
  std::vector<T> m_weights_rec;

  /* LIF dynamics.  */
  T m_beta;
  T m_v_thresh;
  std::vector<T> m_v_membrane;
};

#endif // R_LINEAR_LIF_H_

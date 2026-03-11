#ifndef FR_LINEAR_LIF_H_
#define FR_LINEAR_LIF_H_

#include "layer.hpp"
#include "../tensor.hpp"

/* A fully-connected layer of LIF neurons, with an all-to-all recurrent
   connection (fully recurrent).

   FORNOW: Tracking the output spikes to feed this recurrent connection in
   a thread-safe manner is a challenge; we currently forbid multi-threaded
   access to this layer by insisting on M_BATCH_SIZE==M_NUM_OUTPUTS.  */
class fr_linear_lif : public layer
{
public:
  fr_linear_lif (tensor<float> weights_in, std::vector<float> bias,
		 tensor<float> weights_rec, float beta=0.8f,
		 float v_thresh=1.0f, uint64_t cost=COST_UNDEF);

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin);

  std::vector<uint32_t>
  worstcase_input ();

  void
  reset ();

protected:
  void
  profile_worstcase_batch () override;

private:
  /* Linear parameters.  */
  tensor<float> m_weights_in;
  std::vector<float> m_bias;

  /* Recurrent parameters.  */
  tensor<float> m_weights_rec;

  /* LIF dynamics.  */
  float m_beta;
  float m_v_thresh;
  std::vector<float> m_v_membrane;
  std::vector<uint32_t> m_spike_out;
};

#endif // FR_LINEAR_LIF_H_

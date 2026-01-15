#ifndef LINEAR_LIF_H_
#define LINEAR_LIF_H_

#include <arm_neon.h>
#include "layer.hpp"


class linear_lif : public layer
{
public:

  linear_lif (uint32_t num_inputs, uint32_t num_outputs, float beta=0.8f,
	      float v_thresh=1.0f, uint64_t ts_cost=COST_UNDEF)
    : layer (ts_cost),
      m_num_inputs (num_inputs),
      m_num_outputs (num_outputs),
      m_beta (beta),
      m_v_thresh (v_thresh),
      m_v_membrane (num_outputs)
  {}

  linear_lif (tensor<float> weights, std::vector<float> bias, float beta=0.8f,
	      float v_thresh=1.0f, uint64_t ts_cost=COST_UNDEF)
    : layer (ts_cost),
      m_num_inputs (weights.shape[0]),
      m_num_outputs (weights.shape[1]),
      m_weights (weights),
      m_bias (bias),
      m_beta (beta),
      m_v_thresh (v_thresh),
      m_v_membrane (weights.shape[1])
  {
    assert (weights.shape.size () == 2
	    && weights.shape[1] == bias.size ());
  }

  uint32_t input_size ()  const { return m_num_inputs; }

  uint32_t output_size () const { return m_num_outputs; }

  /* It's difficult to parallelise this update given the binary COO encoded
     SPIKES_IN buffer because NEON lacks gather-loads/scatter-stores.

     The approach taken here is vectorize along the columns of the weight
     matrix, which gives a highly predictable access pattern.  */
  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin, uint32_t batch_end)
  {
    /* Limit before epilogue for a VF of 4.  */
    uint32_t max_neon = batch_end & ~0x03;
    /* Vectorised constants for dynamics.  */
    const float32x4_t beta_splat   = vdupq_n_f32 (m_beta);
    const float32x4_t thresh_splat = vdupq_n_f32 (m_v_thresh);

    /* Handle the neuron's dynamics and linear bias.  */
    float32x4_t membrane, bias;
    for (uint32_t i = 0; i < max_neon; i+=4)
      {
	bias     = vld1q_f32 (&m_bias[i]);
	membrane = vld1q_f32 (&m_v_membrane[i]);

	/* Conditionally set a soft reset value (actually float).  */
	uint32x4_t reset = vandq_u32 (vcgtq_f32 (membrane, thresh_splat),
				      vreinterpretq_u32_f32 (thresh_splat));
	/* Compute bias + (mem * beta) - ?reset.  */
	membrane = vfmaq_f32 (bias, membrane, beta_splat);
	membrane = vsubq_f32 (membrane, vreinterpretq_f32_u32 (reset));

	vst1q_f32 (&m_v_membrane[i], membrane);
      }
    /* Scalar epilogue.  */
    for (uint32_t i = max_neon; i < batch_end; i++)
      {
	float update = m_bias[i] + (m_v_membrane[i] * m_beta);
	if (m_v_membrane[i] > m_v_thresh)
	  update -= m_v_thresh;

	m_v_membrane[i] = update;
      }

    /* Handle the spiking input.  */
    for (uint32_t spike : spikes_in)
      {
	rts_checking_assert (spike < m_weights.shape[0]);
	for (uint32_t i = 0; i < max_neon; i+=4)
	  {
	    float32x4_t weight_slice
	      = vld1q_f32 (&m_weights.vec[i + spike * m_weights.stride[0]]);

	    membrane = vld1q_f32 (&m_v_membrane[i]);
	    membrane = vaddq_f32 (membrane, weight_slice);

	    vst1q_f32 (&m_v_membrane[i], membrane);
	  }
	/* Scalar epilogue.  */
	for (uint32_t i = max_neon; i < batch_end; i++)
	  m_v_membrane[i] += m_weights.vec[i + spike * m_weights.stride[0]];
      }

    std::vector<uint32_t> spike_out;
    /* Push the spiking neurons to SPIKE_OUT.  */
    for (uint32_t i = 0; i < batch_end; i++)
      {
	if (m_v_membrane[i] > m_v_thresh)
	  spike_out.push_back (i);
      }

    return spike_out;
  }

  std::vector<uint32_t>
  timestep (const std::vector<uint32_t> &spikes_in)
  {
    return timestep_batched (spikes_in, 0, m_num_outputs);
  }

  void
  time_worstcase (uint32_t iterations, struct timespec &start,
		  struct timespec &end)
  {
    /* Save the state of any variable dynamics.  */
    std::vector<float> membrane_init = m_v_membrane;

    /* Worst case here is when SPIKES_IN contains all of
       0...(M_NUM_INPUTS-1).  */
    std::vector<uint32_t> spike_in;
    for (uint32_t i = 0; i < m_num_inputs; i++)
      spike_in.push_back (i);

    /* Shuffle this to (hopefully) avoid an optimistically linear
       access pattern.  */
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(spike_in.begin(), spike_in.end(), g);

    clock_gettime (CLOCK_MONOTONIC, &start);
    for (uint32_t i = 0; i < iterations; i++)
      timestep (spike_in);
    clock_gettime (CLOCK_MONOTONIC, &end);

    /* Restore state.  */
    m_v_membrane = membrane_init;
  }

private:
  uint32_t m_num_inputs;
  uint32_t m_num_outputs;

  /* Linear parameters.  */
  tensor<float> m_weights;
  std::vector<float> m_bias;

  /* LIF dynamics.  */
  float m_beta;
  float m_v_thresh;
  std::vector<float> m_v_membrane;
};

#endif // LINEAR_LIF_H_

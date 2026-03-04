#ifndef R_LINEAR_LIF_H_
#define R_LINEAR_LIF_H_

#include <arm_neon.h>
#include <type_traits>
#include "layer.hpp"

/* A fully-connected layer of LIF neurons, each with a one-to-one recurrent
   connection.  */

template <typename T>
class r_linear_lif : public layer
{
public:

  r_linear_lif (tensor<T> weights, std::vector<T> bias, std::vector<T> w_rec,
		uint32_t batch_size, T beta=(T)0.8, T v_thresh=(T)1.0,
		uint64_t batch_cost=COST_UNDEF)
    : layer (weights.shape[0], weights.shape[1], batch_size, batch_cost),
      m_weights (weights),
      m_bias (bias),
      m_weights_rec (w_rec),
      m_beta (beta),
      m_v_thresh (v_thresh),
      m_v_membrane (weights.shape[1])
  {
    static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		   "Invalid type construction for r_linear_lif");

    assert (weights.shape.size () == 2
	    && weights.shape[1] == bias.size ());
    assert (w_rec.size () == weights.shape[1]);
  }

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin, uint32_t batch_end)
  {
    rts_checking_assert (batch_end > batch_begin);

    if constexpr (std::is_same_v<T, float>)
      /* float32_t implementaion.  */
      {
	/* Limit before epilogue for a VF of 4.  */
	uint32_t max_neon = batch_begin + ((batch_end - batch_begin) & ~0x03);
	/* Vectorised constants for dynamics.  */
	const float32x4_t beta_splat   = vdupq_n_f32 (m_beta);
	const float32x4_t thresh_splat = vdupq_n_f32 (m_v_thresh);

	/* Handle the neuron's dynamics, recurrent weight, and linear bias.  */
	float32x4_t membrane, w_rec, bias;
	for (uint32_t i = batch_begin; i < max_neon; i+=4)
	  {
	    bias     = vld1q_f32 (&m_bias[i]);
	    membrane = vld1q_f32 (&m_v_membrane[i]);
	    w_rec    = vld1q_f32 (&m_weights_rec[i]);

	    /* mem > threshold.  */
	    uint32x4_t cmp = vcgtq_f32 (membrane, thresh_splat);
	    /* Conditionally set a soft reset.  */
	    uint32x4_t reset
	      = vandq_u32 (cmp, vreinterpretq_u32_f32 (thresh_splat));
	    /* Conditionally set a recurrent weighting.  */
	    w_rec = vreinterpretq_f32_u32
	      (vandq_u32 (cmp, vreinterpretq_u32_f32 (w_rec)));

	    /* Compute bias + (mem * beta) + w_rec? - reset?  */
	    membrane = vfmaq_f32 (bias, membrane, beta_splat);
	    membrane = vsubq_f32 (membrane, vreinterpretq_f32_u32 (reset));
	    membrane = vaddq_f32 (membrane, w_rec);

	    vst1q_f32 (&m_v_membrane[i], membrane);
	  }
	/* Scalar epilogue.  */
	for (uint32_t i = max_neon; i < batch_end; i++)
	  {
	    float update = m_bias[i] + (m_v_membrane[i] * m_beta);
	    if (m_v_membrane[i] > m_v_thresh)
	      {
		update -= m_v_thresh;
		update += m_weights_rec[i];
	      }
	    m_v_membrane[i] = update;
	  }

	/* Handle the spiking input.  */
	for (uint32_t spike : spikes_in)
	  {
	    rts_checking_assert (spike < m_weights.shape[0]);

	    const uint32_t offset = spike * m_weights.stride[0];
	    for (uint32_t i = batch_begin; i < max_neon; i+=4)
	      {
		float32x4_t weight_slice
		  = vld1q_f32 (&m_weights.vec[offset + i]);

		membrane = vld1q_f32 (&m_v_membrane[i]);
		membrane = vaddq_f32 (membrane, weight_slice);

		vst1q_f32 (&m_v_membrane[i], membrane);
	      }
	    /* Scalar epilogue.  */
	    for (uint32_t i = max_neon; i < batch_end; i++)
	      m_v_membrane[i] += m_weights.vec[offset + i];
	  }
      }
    else
      /* float16_t implementation.  */
      {
	/* Limit before epilogue for a VF of 8.  */
	uint32_t max_neon = batch_begin + ((batch_end - batch_begin) & ~0x07);
	/* Vectorised constants for dynamics.  */
	const float16x8_t beta_splat   = vdupq_n_f16 (m_beta);
	const float16x8_t thresh_splat = vdupq_n_f16 (m_v_thresh);

	/* Handle the neuron's dynamics, recurrent weight, and linear bias.  */
	float16x8_t membrane, w_rec, bias;
	for (uint32_t i = batch_begin; i < max_neon; i+=8)
	  {
	    bias     = vld1q_f16 (&m_bias[i]);
	    membrane = vld1q_f16 (&m_v_membrane[i]);
	    w_rec    = vld1q_f16 (&m_weights_rec[i]);

	    /* mem > threshold.  */
	    uint16x8_t cmp = vcgtq_f16 (membrane, thresh_splat);
	    /* Conditionally set a soft reset.  */
	    uint16x8_t reset
	      = vandq_u16 (cmp, vreinterpretq_u16_f16 (thresh_splat));
	    /* Conditionally set a recurrent weighting.  */
	    w_rec = vreinterpretq_f16_u16
	      (vandq_u16 (cmp, vreinterpretq_u16_f16 (w_rec)));

	    /* Compute bias + (mem * beta) + w_rec? - reset?  */
	    membrane = vfmaq_f16 (bias, membrane, beta_splat);
	    membrane = vsubq_f16 (membrane, vreinterpretq_f16_u16 (reset));
	    membrane = vaddq_f16 (membrane, w_rec);

	    vst1q_f16 (&m_v_membrane[i], membrane);
	  }
	/* Scalar epilogue.  */
	for (uint32_t i = max_neon; i < batch_end; i++)
	  {
	    float16_t update = m_bias[i] + (m_v_membrane[i] * m_beta);
	    if (m_v_membrane[i] > m_v_thresh)
	      {
		update -= m_v_thresh;
		update += m_weights_rec[i];
	      }
	    m_v_membrane[i] = update;
	  }
	/* Handle the spiking input.  */
	for (uint32_t spike : spikes_in)
	  {
	    rts_checking_assert (spike < m_weights.shape[0]);

	    const uint32_t offset = spike * m_weights.stride[0];
	    for (uint32_t i = batch_begin; i < max_neon; i+=8)
	      {
		float16x8_t weight_slice
		  = vld1q_f16 (&m_weights.vec[offset + i]);

		membrane = vld1q_f16 (&m_v_membrane[i]);
		membrane = vaddq_f16 (membrane, weight_slice);

		vst1q_f16 (&m_v_membrane[i], membrane);
	      }
	    /* Scalar epilogue.  */
	    for (uint32_t i = max_neon; i < batch_end; i++)
	      m_v_membrane[i] += m_weights.vec[offset + i];
	  }
      }

    std::vector<uint32_t> spike_out;
    /* Push the spiking neurons to SPIKE_OUT.  */
    for (uint32_t i = batch_begin; i < batch_end; i++)
      {
	if (m_v_membrane[i] > m_v_thresh)
	  spike_out.push_back (i);
      }

    return spike_out;
  }

  uint64_t
  time_batch_worstcase_ns ()
  {
    /* Save the state of any variable dynamics.  */
    std::vector<T> membrane_init = m_v_membrane;

    /* Worst case here is when SPIKES_IN contains all of
       0...(M_NUM_INPUTS-1).  */
    std::vector<uint32_t> spike_in;
    for (uint32_t i = 0; i < m_num_inputs; i++)
      spike_in.push_back (i);

    /* Shuffle this to (hopefully) avoid an optimistically linear
       access pattern.  */
    std::random_device rd;
    std::mt19937 g (rd ());
    std::shuffle (spike_in.begin (), spike_in.end (), g);

    timespec start, end;
    clock_gettime (CLOCK_MONOTONIC, &start);
    timestep_batched (spike_in, 0, m_batch_size);
    clock_gettime (CLOCK_MONOTONIC, &end);

    /* Restore state.  */
    m_v_membrane = membrane_init;

    return (end.tv_sec - start.tv_sec) * 1E9
      + end.tv_nsec - start.tv_nsec;
  }

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

#include <random>
#include <algorithm>
#include <arm_neon.h>
#include "../tensor.hpp"
#include "../util.h"
#include "fr_linear_lif.hpp"

fr_linear_lif::fr_linear_lif (tensor<float> weights_in, std::vector<float> bias,
			      tensor<float> weights_rec, float beta,
			      float v_thresh, uint64_t cost)
  : layer (weights_in.shape[0], weights_in.shape[1], weights_in.shape[1], cost,
	   "FR_LLIF"),
    m_weights_in (weights_in),
    m_bias (bias),
    m_weights_rec (weights_rec),
    m_beta (beta),
    m_v_thresh (v_thresh),
    m_v_membrane (weights_in.shape[1])
{
  assert (weights_in.shape.size () == 2 && weights_in.shape[1] == bias.size ());

  assert (weights_rec.shape.size () == 2
	  && weights_rec.shape[0] == weights_rec.shape[1]);

  assert (weights_rec.shape[0] == weights_in.shape[1]);
}

std::vector<uint32_t>
fr_linear_lif::timestep_batched (const std::vector<uint32_t> &spikes_in,
				 uint32_t batch_begin, uint32_t batch_end)
{
  rts_checking_assert (batch_end > batch_begin);
  /* Limit before epilogue for a VF of 4.  */
  uint32_t max_neon = batch_begin + ((batch_end - batch_begin) & ~0x03);
  /* Vectorised constants for dynamics.  */
  const float32x4_t beta_splat   = vdupq_n_f32 (m_beta);
  const float32x4_t thresh_splat = vdupq_n_f32 (m_v_thresh);

    /* Handle the neuron's dynamics and linear bias.  */
  float32x4_t membrane, bias;
  for (uint32_t i = batch_begin; i < max_neon; i+=4)
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

  /* Handle the input spikes.  */
  for (uint32_t spike : spikes_in)
    {
      rts_checking_assert (spike < m_weights_in.shape[0]);

      const uint32_t offset = spike * m_weights_in.stride[0];
      for (uint32_t i = batch_begin; i < max_neon; i+=4)
	{
	  float32x4_t weight_slice
	    = vld1q_f32 (&m_weights_in.vec[offset + i]);

	  membrane = vld1q_f32 (&m_v_membrane[i]);
	  membrane = vaddq_f32 (membrane, weight_slice);

	  vst1q_f32 (&m_v_membrane[i], membrane);
	}
      /* Scalar epilogue.  */
      for (uint32_t i = max_neon; i < batch_end; i++)
	m_v_membrane[i] += m_weights_in.vec[offset + i];
    }

  /* Handle the recurrent spikes.  */
  for (uint32_t spike : m_spike_out)
    {
      rts_checking_assert (spike < m_weights_rec.shape[0]);

      const uint32_t offset = spike * m_weights_rec.stride[0];
      for (uint32_t i = batch_begin; i < max_neon; i+=4)
	{
	  float32x4_t weight_slice
	    = vld1q_f32 (&m_weights_rec.vec[offset + i]);

	  membrane = vld1q_f32 (&m_v_membrane[i]);
	  membrane = vaddq_f32 (membrane, weight_slice);

	  vst1q_f32 (&m_v_membrane[i], membrane);
	}
      /* Scalar epilogue.  */
      for (uint32_t i = max_neon; i < batch_end; i++)
	m_v_membrane[i] += m_weights_rec.vec[offset + i];
    }

  /* NOTE: This is clearly not thread-safe, but we've guaranteed that this
     layer is processed by at most one thread (see constructor).  */
  m_spike_out.clear ();
  /* Push the spiking neurons to M_SPIKE_OUT.  */
  for (uint32_t i = batch_begin; i < batch_end; i++)
    {
      if (m_v_membrane[i] > m_v_thresh)
	m_spike_out.push_back (i);
    }

  return m_spike_out;
}

std::vector<uint32_t>
fr_linear_lif::worstcase_input ()
{
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

  return spike_in;
}

void
fr_linear_lif::reset ()
{
  /* Reset membrane potentials to zero.  */
  std::fill (m_v_membrane.begin (), m_v_membrane.end (), 0);
  /* Reset spikes out to empty.   */
  m_spike_out.clear ();
}

void
fr_linear_lif::profile_worstcase_batch ()
{
  timespec start, end;
  std::vector<uint32_t> input = worstcase_input ();

  /* We must also consider recurrent spikes here.  */
  for (uint32_t i = 0; i < m_num_outputs; i++)
    m_spike_out.push_back (i);

  /* Measure the execution time under the heaviest load.   */
  clock_gettime (CLOCK_MONOTONIC, &start);
  timestep_batched (input, 0, m_batch_size);
  clock_gettime (CLOCK_MONOTONIC, &end);

  m_batch_cost_ns = (end.tv_sec - start.tv_sec) * 1E9
    + end.tv_nsec - start.tv_nsec;

  /* Reset state.  */
  reset ();
}

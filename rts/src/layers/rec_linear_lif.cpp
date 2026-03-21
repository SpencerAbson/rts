#include <arm_neon.h>
#include <algorithm>
#include <type_traits>
#include "../../include/util.h"
#include "../../include/tensor.hpp"
#include "../../include/layers/rec_linear_lif.hpp"

/* NOTE: Dervied members are not automatically brought into scope by
   C++'s templated inheritance rules, hence 'this->'.  */

template<typename T>
rec_linear_lif<T>::rec_linear_lif (tensor<T> weights, std::vector<T> bias,
				   std::vector<T> w_rec, uint32_t batch_size,
				   T beta, T v_thresh, uint64_t batch_cost)
  : linear_lif<T> (weights, bias, batch_size, beta, v_thresh, batch_cost,
		   "REC_LLIF"),
    m_weights_rec (w_rec)
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for rec_linear_lif");

  assert (w_rec.size () == weights.shape[1]);
}

template<typename T>
rec_linear_lif<T>::rec_linear_lif
	(const std::string &path_w, const std::string &path_b,
	 const std::string &path_rw, uint32_t num_inputs, uint32_t num_outputs,
	 uint32_t batch_size, T beta, T v_thresh, uint64_t batch_cost)
  : linear_lif<T> (path_w, path_b, num_inputs, num_outputs, batch_size, beta,
		   v_thresh, batch_cost, "REC_LLIF")
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for rec_linear_lif");

  int res = weights_from_file (path_rw, num_outputs, m_weights_rec);
  assert (!res && "Failed to read recurrent weights.");
}

template<typename T>
void
rec_linear_lif<T>::f32_neuron_update (uint32_t batch_begin, uint32_t batch_end)
{
  if constexpr (std::is_same_v<T, float32_t>)
    {
      /* Max iteration at VF of 4.  */
      const uint32_t vector_max
	= batch_begin + ((batch_end - batch_begin) & ~0x03);
      /* Vectorised constants for dynamics.  */
      const float32x4_t beta_splat   = vdupq_n_f32 (this->m_beta);
      const float32x4_t thresh_splat = vdupq_n_f32 (this->m_v_thresh);

      /* Handle the neuron's dynamics, recurrent weight, and linear bias.  */
      float32x4_t membrane, w_rec, bias;
      for (uint32_t i = batch_begin; i < vector_max; i+=4)
	{
	  bias     = vld1q_f32 (&this->m_bias[i]);
	  membrane = vld1q_f32 (&this->m_v_membrane[i]);
	  w_rec    = vld1q_f32 (&this->m_weights_rec[i]);

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

	  vst1q_f32 (&this->m_v_membrane[i], membrane);
	}
      /* Scalar epilogue.  */
      for (uint32_t i = vector_max; i < batch_end; i++)
	{
	  float update = this->m_bias[i]
	    + (this->m_v_membrane[i] * this->m_beta);
	  if (this->m_v_membrane[i] > this->m_v_thresh)
	    {
	      update -= this->m_v_thresh;
	      update += this->m_weights_rec[i];
	    }
	  this->m_v_membrane[i] = update;
	}
    }
}

template<typename T>
void
rec_linear_lif<T>::f16_neuron_update (uint32_t batch_begin, uint32_t batch_end)
{
  if constexpr (std::is_same_v<T, float16_t>)
    {
      /* Max iteration at VF of 8.  */
      const uint32_t vector_max
	= batch_begin + ((batch_end - batch_begin) & ~0x07);
      /* Vectorised constants for dynamics.  */
      const float16x8_t beta_splat   = vdupq_n_f16 (this->m_beta);
      const float16x8_t thresh_splat = vdupq_n_f16 (this->m_v_thresh);

      /* Handle the neuron's dynamics, recurrent weight, and linear bias.  */
      float16x8_t membrane, w_rec, bias;
      for (uint32_t i = batch_begin; i < vector_max; i+=8)
	{
	  bias     = vld1q_f16 (&this->m_bias[i]);
	  membrane = vld1q_f16 (&this->m_v_membrane[i]);
	  w_rec    = vld1q_f16 (&this->m_weights_rec[i]);

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

	  vst1q_f16 (&this->m_v_membrane[i], membrane);
	}
      /* Scalar epilogue.  */
      for (uint32_t i = vector_max; i < batch_end; i++)
	{
	  float16_t update = this->m_bias[i]
	    + (this->m_v_membrane[i] * this->m_beta);
	  if (this->m_v_membrane[i] > this->m_v_thresh)
	    {
	      update -= this->m_v_thresh;
	      update += this->m_weights_rec[i];
	    }
	  this->m_v_membrane[i] = update;
	}
    }
}

template<typename T>
void
rec_linear_lif<T>::profile_worstcase_batch ()
{
  timespec start, end;
  std::vector<uint32_t> input = this->worstcase_input ();

  /* We must also consider recurrent spikes here, so set the
     current membrane potential to something over the threshold.  */
  std::fill (this->m_v_membrane.begin (), this->m_v_membrane.end (),
	     this->m_v_thresh + 1);

  /* Measure the execution time under the heaviest load.  */
  clock_gettime (CLOCK_MONOTONIC, &start);
  this->timestep_batched (input, 0, this->m_batch_size);
  clock_gettime (CLOCK_MONOTONIC, &end);

  this->m_batch_cost_ns = (end.tv_sec - start.tv_sec) * 1E9
    + end.tv_nsec - start.tv_nsec;

  /* Reset state.  */
  this->reset ();
}

template class rec_linear_lif<float32_t>;
template class rec_linear_lif<float16_t>;

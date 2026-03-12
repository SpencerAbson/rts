#include <random>
#include <string>
#include <arm_neon.h>
#include <type_traits>
#include "../tensor.hpp"
#include "../util.h"
#include "linear_lif.hpp"

template<typename T>
linear_lif<T>::linear_lif (tensor<T> weights, std::vector<T> bias,
			   uint32_t batch_size, T beta, T v_thresh,
			   uint64_t batch_cost, std::string type)
  : layer (weights.shape[0], weights.shape[1], batch_size, batch_cost, type),
    m_weights (weights),
    m_bias (bias),
    m_beta (beta),
    m_v_thresh (v_thresh),
    m_v_membrane (weights.shape[1])
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for linear_lif");

  assert (weights.shape.size () == 2 && weights.shape[1] == bias.size ());
}

template<typename T>
std::vector<uint32_t>
linear_lif<T>::timestep_batched (const std::vector<uint32_t> &spikes_in,
				 uint32_t batch_begin, uint32_t batch_end)
{
  rts_checking_assert (batch_begin < batch_end);
  if constexpr (std::is_same_v<T, float>)
    /* float32_t implementaion.  */
    {
      f32_neuron_update (batch_begin, batch_end);
      f32_spike_prop (spikes_in, batch_begin, batch_end);
    }
  else if constexpr (std::is_same_v<T, float16_t>)
    /* float16_t implementation.  */
    {
      f16_neuron_update (batch_begin, batch_end);
      f16_spike_prop (spikes_in, batch_begin, batch_end);
    }
  else
    rts_unreachable ("Type construction for linear_lif");

  std::vector<uint32_t> spike_out;
  /* Push the spiking neurons to SPIKE_OUT.  */
  for (uint32_t i = batch_begin; i < batch_end; i++)
    {
      if (m_v_membrane[i] > m_v_thresh)
	  spike_out.push_back (i);
    }

  return spike_out;
}

template<typename T>
void
linear_lif<T>::f32_neuron_update (uint32_t batch_begin, uint32_t batch_end)
{
 if constexpr (std::is_same_v<T, float32_t>)
   {
     /* Max iteration at VF of 4.  */
     const uint32_t vector_max
       = batch_begin + ((batch_end - batch_begin) & ~0x03);
     /* Vectorised constants for dynamics.  */
     const float32x4_t beta_splat   = vdupq_n_f32 (m_beta);
     const float32x4_t thresh_splat = vdupq_n_f32 (m_v_thresh);

     /* Handle the neuron's dynamics and linear bias.  */
     float32x4_t membrane, bias;
     for (uint32_t i = batch_begin; i < vector_max; i+=4)
       {
	 bias     = vld1q_f32 (&m_bias[i]);
	 membrane = vld1q_f32 (&m_v_membrane[i]);

	 /* Conditionally set a soft reset value.  */
	 uint32x4_t reset = vandq_u32 (vcgtq_f32 (membrane, thresh_splat),
				       vreinterpretq_u32_f32 (thresh_splat));
	 /* Compute bias + (mem * beta) - ?reset.  */
	 membrane = vfmaq_f32 (bias, membrane, beta_splat);
	 membrane = vsubq_f32 (membrane, vreinterpretq_f32_u32 (reset));
	 vst1q_f32 (&m_v_membrane[i], membrane);
       }
     /* Scalar epilogue.  */
     for (uint32_t i = vector_max; i < batch_end; i++)
       {
	 float update = m_bias[i] + (m_v_membrane[i] * m_beta);
	 if (m_v_membrane[i] > m_v_thresh)
	   update -= m_v_thresh;

	 m_v_membrane[i] = update;
       }
   }
 else
   {
     rts_unreachable ("Type construction for linear_lif");
   }
}

template<typename T>
void
linear_lif<T>::f16_neuron_update (uint32_t batch_begin, uint32_t batch_end)
{
 if constexpr (std::is_same_v<T, float16_t>)
   {
     /* Max iteration at VF of 8.  */
     const uint32_t vector_max
       = batch_begin + ((batch_end - batch_begin) & ~0x07);
     /* Vectorised constants for dynamics.  */
     const float16x8_t beta_splat   = vdupq_n_f16 (m_beta);
     const float16x8_t thresh_splat = vdupq_n_f16 (m_v_thresh);

     /* Handle the neuron's dynamics and linear bias.  */
     float16x8_t membrane, bias;
     for (uint32_t i = batch_begin; i < vector_max; i+=8)
       {
	 bias     = vld1q_f16 (&m_bias[i]);
	 membrane = vld1q_f16 (&m_v_membrane[i]);

	 /* Conditionally set a soft reset value.  */
	 uint16x8_t reset = vandq_u16 (vcgtq_f16 (membrane, thresh_splat),
				       vreinterpretq_u16_f16 (thresh_splat));
	 /* Compute bias + (mem * beta) - ?reset.  */
	 membrane = vfmaq_f16 (bias, membrane, beta_splat);
	 membrane = vsubq_f16 (membrane, vreinterpretq_f16_u16 (reset));

	 vst1q_f16 (&m_v_membrane[i], membrane);
       }
     /* Scalar epilogue.  */
     for (uint32_t i = vector_max; i < batch_end; i++)
       {
	 float16_t update = m_bias[i] + (m_v_membrane[i] * m_beta);
	 if (m_v_membrane[i] > m_v_thresh)
	   update -= m_v_thresh;

	 m_v_membrane[i] = update;
       }
   }
 else
   {
     rts_unreachable ("Type construction for linear_lif");
   }
}

template<typename T>
void
linear_lif<T>::f32_spike_prop (const std::vector<uint32_t> &spikes_in,
			       uint32_t batch_begin, uint32_t batch_end)
{
  if constexpr (std::is_same_v<T, float32_t>)
    {
      /* Max unrolled and vectorised iterations.  */
      const uint32_t unroll_max
	= batch_begin + ((batch_end - batch_begin) & ~0x0F);
      const uint32_t vector_max
	= batch_begin + ((batch_end - batch_begin) & ~0x03);

      if (spikes_in.size () != 0)
	{
	  uint32_t stride = m_weights.stride[0];
	  uint32_t offset;
	  uint32_t next_offset = spikes_in[0] * stride;
	  for (uint32_t i = 0; i < spikes_in.size () - 1; i++)
	    {
	      offset = next_offset;
	      /* We'll manually prefetch the next row of weights whilst
		 working on this one.  */
	      next_offset = spikes_in[i + 1] * stride;
	      /* PRFM instructions are not cheap, so we'd like to minimise our
		 use of them as much as possible.  We expect the memory system
		 to preload the 64-byte cache line pointed to by each PRFM
		 instruction into L2, so we need to use at most one per 64-bytes
		 (16 floats).

		 At the same time, we would like to avoid code like:

		 if ((counter & 0x0F) == 0)
		   __builtin_prefetch (...)

		 from appearing in the loop, since it would introduce branch
		 mis-predictions and slow us down.  The easy-route taken here
		 is to apply an unrolling factor of 16, and emit one PFRM
		 instruction  per iteration.  */
	      for (uint32_t j = batch_begin; j < unroll_max; j+= 16)
		{
		  __builtin_prefetch (&m_weights.vec[next_offset + j], 0, 2);

		  float32x4_t weights
		    = vld1q_f32 (&m_weights.vec[offset + j]);
		  float32x4_t membranes
		    = vld1q_f32 (&m_v_membrane[j]);
		  vst1q_f32 (&m_v_membrane[j],
			     vaddq_f32 (membranes, weights));

		  weights = vld1q_f32 (&m_weights.vec[offset + j + 4]);
		  membranes = vld1q_f32 (&m_v_membrane[j + 4]);
		  vst1q_f32 (&m_v_membrane[j + 4],
			     vaddq_f32 (membranes, weights));

		  weights = vld1q_f32 (&m_weights.vec[offset + j + 8]);
		  membranes = vld1q_f32 (&m_v_membrane[j + 8]);
		  vst1q_f32 (&m_v_membrane[j + 8],
			     vaddq_f32 (membranes, weights));

		  weights = vld1q_f32 (&m_weights.vec[offset + j + 12]);
		  membranes = vld1q_f32 (&m_v_membrane[j + 12]);
		  vst1q_f32 (&m_v_membrane[j + 12],
			     vaddq_f32 (membranes, weights));
		}
	      /* Unroll epilogue.  */
	      for (uint32_t j = unroll_max; j < vector_max; j+=4)
		{
		  float32x4_t weights
		    = vld1q_f32 (&m_weights.vec[offset + j]);
		  float32x4_t membranes
		    = vld1q_f32 (&m_v_membrane[j]);

		  membranes = vaddq_f32 (membranes, weights);
		  vst1q_f32 (&m_v_membrane[j], membranes);
		}
	      /* Vector epilogue.  */
	      for (uint32_t j = vector_max; j < batch_end; j++)
		m_v_membrane[j] += m_weights.vec[offset + j];
	    }
	  /* The final spike...  */
	  for (uint32_t j = batch_begin; j < unroll_max; j+= 16)
	    {
	      float32x4_t weights
		= vld1q_f32 (&m_weights.vec[next_offset + j]);
	      float32x4_t membranes
		= vld1q_f32 (&m_v_membrane[j]);
	      vst1q_f32 (&m_v_membrane[j],
			 vaddq_f32 (membranes, weights));

	      weights = vld1q_f32 (&m_weights.vec[next_offset + j + 4]);
	      membranes = vld1q_f32 (&m_v_membrane[j + 4]);
	      vst1q_f32 (&m_v_membrane[j + 4],
			 vaddq_f32 (membranes, weights));

	      weights = vld1q_f32 (&m_weights.vec[next_offset + j + 8]);
	      membranes = vld1q_f32 (&m_v_membrane[j + 8]);
	      vst1q_f32 (&m_v_membrane[j + 8],
			 vaddq_f32 (membranes, weights));

	      weights = vld1q_f32 (&m_weights.vec[next_offset + j + 12]);
	      membranes = vld1q_f32 (&m_v_membrane[j + 12]);
	      vst1q_f32 (&m_v_membrane[j + 12],
			 vaddq_f32 (membranes, weights));
	    }
	  /* Unroll epilogue.  */
	  for (uint32_t j = unroll_max; j < vector_max; j+=4)
	    {
	      float32x4_t weights
		= vld1q_f32 (&m_weights.vec[next_offset + j]);
	      float32x4_t membranes
		= vld1q_f32 (&m_v_membrane[j]);

	      membranes = vaddq_f32 (membranes, weights);
	      vst1q_f32 (&m_v_membrane[j], membranes);
	    }
	  /* Vector epilogue.  */
	  for (uint32_t j = vector_max; j < batch_end; j++)
	    m_v_membrane[j] += m_weights.vec[next_offset + j];
	}
    }
  else
    {
      rts_unreachable ("Type construction for linear_lif");
    }
}

template<typename T>
void
linear_lif<T>::f16_spike_prop (const std::vector<uint32_t> &spikes_in,
			       uint32_t batch_begin, uint32_t batch_end)
{
  if constexpr (std::is_same_v<T, float16_t>)
    {
      /* Max unrolled and vectorised iterations.  */
      const uint32_t unroll_max
	= batch_begin + ((batch_end - batch_begin) & ~0x1F);
      const uint32_t vector_max
	= batch_begin + ((batch_end - batch_begin) & ~0x07);

      if (spikes_in.size () != 0)
	{
	  uint32_t stride = m_weights.stride[0];
	  uint32_t offset;
	  uint32_t next_offset = spikes_in[0] * stride;
	  for (uint32_t i = 0; i < spikes_in.size () - 1; i++)
	    {
	      offset = next_offset;
	      /* We'll manually prefetch the next row of weights whilst
		 working on this one.  */
	      next_offset = spikes_in[i + 1] * stride;
	      /* PRFM instructions are not cheap, so we'd like to minimise our
		 use of them as much as possible.  We expect the memory system
		 to preload the 64-byte cache line pointed to by each PRFM
		 instruction into L2, so we need to use at most one per 64-bytes
		 (32 halfs).

		 At the same time, we would like to avoid code like:

		 if ((counter & 0x1F) == 0)
		   __builtin_prefetch (...)

		 from appearing in the loop, since it would introduce branch
		 mis-predictions and slow us down.  The easy-route taken here
		 is to apply an unrolling factor of 32, and emit one PFRM
		 instruction  per iteration.  */
	      for (uint32_t j = batch_begin; j < unroll_max; j+= 32)
		{
		  __builtin_prefetch (&m_weights.vec[next_offset + j], 0, 2);

		  float16x8_t weights
		    = vld1q_f16 (&m_weights.vec[offset + j]);
		  float16x8_t membranes
		    = vld1q_f16 (&m_v_membrane[j]);
		  vst1q_f16 (&m_v_membrane[j],
			     vaddq_f16 (membranes, weights));

		  weights = vld1q_f16 (&m_weights.vec[offset + j + 8]);
		  membranes = vld1q_f16 (&m_v_membrane[j + 8]);
		  vst1q_f16 (&m_v_membrane[j + 8],
			     vaddq_f16 (membranes, weights));

		  weights = vld1q_f16 (&m_weights.vec[offset + j + 16]);
		  membranes = vld1q_f16 (&m_v_membrane[j + 16]);
		  vst1q_f16 (&m_v_membrane[j + 16],
			     vaddq_f16 (membranes, weights));

		  weights = vld1q_f16 (&m_weights.vec[offset + j + 24]);
		  membranes = vld1q_f16 (&m_v_membrane[j + 24]);
		  vst1q_f16 (&m_v_membrane[j + 24],
			     vaddq_f16 (membranes, weights));
		}
	      /* Unroll epilogue.  */
	      for (uint32_t j = unroll_max; j < vector_max; j+=8)
		{
		  float16x8_t weights
		    = vld1q_f16 (&m_weights.vec[offset + j]);
		  float16x8_t membranes
		    = vld1q_f16 (&m_v_membrane[j]);

		  membranes = vaddq_f16 (membranes, weights);
		  vst1q_f16 (&m_v_membrane[j], membranes);
		}
	      /* Vector epilogue.  */
	      for (uint32_t j = vector_max; j < batch_end; j++)
		m_v_membrane[j] += m_weights.vec[offset + j];
	    }
	  /* The final spike...  */
	  for (uint32_t j = batch_begin; j < unroll_max; j+= 32)
	    {
	      float16x8_t weights
		= vld1q_f16 (&m_weights.vec[next_offset + j]);
	      float16x8_t membranes
		= vld1q_f16 (&m_v_membrane[j]);
	      vst1q_f16 (&m_v_membrane[j],
			 vaddq_f16 (membranes, weights));

	      weights = vld1q_f16 (&m_weights.vec[next_offset + j + 8]);
	      membranes = vld1q_f16 (&m_v_membrane[j + 8]);
	      vst1q_f16 (&m_v_membrane[j + 8],
			 vaddq_f16 (membranes, weights));

	      weights = vld1q_f16 (&m_weights.vec[next_offset + j + 16]);
	      membranes = vld1q_f16 (&m_v_membrane[j + 16]);
	      vst1q_f16 (&m_v_membrane[j + 16],
			 vaddq_f16 (membranes, weights));

	      weights = vld1q_f16 (&m_weights.vec[next_offset + j + 24]);
	      membranes = vld1q_f16 (&m_v_membrane[j + 24]);
	      vst1q_f16 (&m_v_membrane[j + 24],
			 vaddq_f16 (membranes, weights));
	    }
	  /* Unroll epilogue.  */
	  for (uint32_t j = unroll_max; j < vector_max; j+=8)
	    {
	      float16x8_t weights
		= vld1q_f16 (&m_weights.vec[next_offset + j]);
	      float16x8_t membranes
		= vld1q_f16 (&m_v_membrane[j]);

	      membranes = vaddq_f16 (membranes, weights);
	      vst1q_f16 (&m_v_membrane[j], membranes);
	    }
	  /* Vector epilogue.  */
	  for (uint32_t j = vector_max; j < batch_end; j++)
	    m_v_membrane[j] += m_weights.vec[next_offset + j];
	}
    }
  else
    {
      rts_unreachable ("Type construction for linear_lif");
    }
}

template<typename T>
std::vector<uint32_t>
linear_lif<T>::worstcase_input ()
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

template<typename T>
void
linear_lif<T>::reset ()
{
  /* Reset membrane potentials to zero.  */
  std::fill (m_v_membrane.begin (), m_v_membrane.end (), (T)0);
}

template class linear_lif<float32_t>;
template class linear_lif<float16_t>;

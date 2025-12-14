#ifndef LINEAR_LIF_H_
#define LINEAR_LIF_H_

#include <arm_neon.h>
#include "layer.hpp"


class linear_lif : public layer
{
public:

  linear_lif (uint32_t num_inputs, uint32_t num_outputs, float beta=0.8f,
	      float v_thresh=1.0f)
    : num_inputs (num_inputs),
      num_outputs (num_outputs),
      beta (beta),
      v_thresh (v_thresh),
      v_membrane (num_outputs)
  {}

  linear_lif (tensor<float> weights, std::vector<float> bias, float beta=0.8f,
	      float v_thresh=1.0f)
    : num_inputs (weights.shape[0]),
      num_outputs (weights.shape[1]),
      weights (weights),
      bias (bias),
      beta (beta),
      v_thresh (v_thresh),
      v_membrane (weights.shape[1])
  {
    assert (weights.shape.size () == 2
	    && weights.shape[1] == bias.size ());
  }

  uint32_t input_size ()  const { return num_inputs; }

  uint32_t output_size () const { return num_outputs; }

  uint32_t timestep_cost () const { return num_inputs * num_outputs; }

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin, uint32_t batch_end)
  {
    std::vector<uint32_t> spikes_out;

    float32x4_t beta_splat = vdupq_n_f32 (beta);
    float32x4_t thresh_splat = vdupq_n_f32 (v_thresh);

    /* It's difficult to parallelise this update given the binary COO encoded
       input buffer because NEON lacks gather-loads/scatter-stores.

       The approach taken here is vectorize along the columns of the weight
       matrix (and therefore the layer size), linearly processing each input
       spike across a 'slice' of values.

       A minor headache is that we have to scalarize the conditional pushes
       to the output buffer, but this seems to incur negligable overhead.  */
    uint32_t max_neon = batch_end & ~0x03;
    for (uint32_t i = batch_begin; i < max_neon; i+=4)
      {
	/* SLICE holds one value for each column that we're processing.  */
	float32x4_t slice = vld1q_f32 (&bias[i]);
	for (uint32_t j  = 0; j < spikes_in.size (); j++)
	  {
	    rts_checking_assert (spikes_in[j] < weights.shape[0]);
	    float32x4_t weight_slice
	      = vld1q_f32 (&weights.vec[i + spikes_in[j] * weights.stride[0]]);

	    slice = vaddq_f32 (slice, weight_slice);
	  }

	/* The membrane potentials of the the neurons associated with each
	   value in SLICE.  */
	float32x4_t membrane = vld1q_f32 (&v_membrane[i]);
	/* Conditionally set a soft reset value (actually float).  */
	uint32x4_t reset = vandq_u32 (vcgtq_f32 (membrane, thresh_splat),
				      vreinterpretq_u32_f32 (thresh_splat));

	/* Compute (mem * beta) + weighted_spk_in - ?reset.  */
	slice = vfmaq_f32 (slice, membrane, beta_splat);
	slice = vsubq_f32 (slice, vreinterpretq_f32_u32 (reset));

	/* Compare the new potentials to V_THRESH.  */
	uint32x4_t cmp = vcgtq_f32 (slice, thresh_splat);
	/* Now scalarize the conditional pushes to SPIKES_OUT.  This code
	   is verbose because the 'lane' arguments to vget/vset must be
	   integer literals, but I don't think it warrants a macro.  */
	if (vgetq_lane_u32 (cmp, 0) != 0)
	  spikes_out.push_back (i);
	if (vgetq_lane_u32 (cmp, 1) != 0)
	  spikes_out.push_back (i+1);
	if (vgetq_lane_u32 (cmp, 2) != 0)
	  spikes_out.push_back (i+2);
	if (vgetq_lane_u32 (cmp, 3) != 0)
	  spikes_out.push_back (i+3);

	vst1q_f32 (&v_membrane[i], slice);
      }
    /* Scalar epilogue.  */
    for (uint32_t i = max_neon; i < batch_end; i++)
      {
	float update = bias[i];
	for (uint32_t j = 0; j < spikes_in.size (); j++)
	  {
	    rts_checking_assert (spikes_in[j] < weights.shape[0]);
	    update += weights.vec[i + spikes_in[j] * weights.stride[0]];
	  }

	update = v_membrane[i] * beta + update;
	if (v_membrane[i] > v_thresh)
	  update -= v_thresh;

	if (update > v_thresh)
	  spikes_out.push_back (i);

	v_membrane[i] = update;
      }

    return spikes_out;
  }

  std::vector<uint32_t>
  timestep (const std::vector<uint32_t> &spikes_in)
  {
    return timestep_batched (spikes_in, 0, weights.shape[1]);
  }

private:

  uint32_t num_inputs;
  uint32_t num_outputs;

  /* Linear parameters.  */
  tensor<float> weights;
  std::vector<float> bias;

  /* LIF dynamics.  */
  float beta;
  float v_thresh;
  std::vector<float> v_membrane;
};

#endif // LINEAR_LIF_H_

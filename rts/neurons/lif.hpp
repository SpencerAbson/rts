#ifndef LIF_H_
#define LIF_H_

#include <arm_neon.h>
#include "../tensor.hpp"

/* Linearly-weighted layer of SIZE leaky integrate-and-fire neurons.

   We'll probably want to bundle the weights and biases into this class,
   but it's convenient to parameterise them for early testing.  */
class linear_lif
{
public:
  linear_lif (uint32_t size, float beta=0.8f, float v_thresh=1.0f)
    : beta (beta), v_thresh (v_thresh), v_membrane (size)
  {};

  /* Simulate the layer for one timestep using WEIGHTS and SPIKES_IN.

     Each i in SPIKES_IN represents a spike from neuron_i in the previous
     layer at the previous timestep.  This representation of the input
     vector is somewhat related to the COO representation of sparse matrices,
     though in this case we're handling binary information and can omit the
     'value' part.

     SPIKES_OUT is an identically encoded vector of output spikes from this
     layer and at the previous timestep.  */
  std::vector<uint32_t>
  forward (const std::vector<uint32_t> &spikes_in,
	   const tensor<float> &weights, const std::vector<float> &bias)
  {
    /* We're expecing WEIGHTS to be a maxtrix with a column of input weights
       for each neuron in this layer.  */
    assert (weights.shape.size () == 2
	    && weights.shape[1] == v_membrane.size ());
    /* Similarly, we expect one BIAS value per neuron.  */
    assert (bias.size () == v_membrane.size ());

    std::vector<uint32_t> spikes_out;
    float32x4_t beta_splat = vdupq_n_f32 (beta);
    float32x4_t thresh_splat = vdupq_n_f32 (v_thresh);

    /* It's difficult to parallelise this update given the binary COO encoded
       SPIKES_IN vector because NEON lacks gather-loads/scatter-stores.

       The approach taken here is vectorize along the columns of the weight
       matrix (and therefore the layer size), linearly processing each
       SPIKE_IN across a 'slice' of values.

       A minor headache is that we have to scalarize the conditional pushes
       to SPIKES_OUT, but this seems to incur negligable overhead.  */
    uint32_t max_neon = weights.shape[1] & ~0x03;
    for (uint32_t i = 0; i < max_neon; i+=4)
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
    for (uint32_t i = max_neon; i < weights.shape[1]; i++)
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

  /* Purely scalar implementation of the above.  */
  std::vector<uint32_t>
  forward_scalar (const std::vector<uint32_t> &spikes_in,
		  const tensor<float> &weights, const std::vector<float> bias)
  {
    /* We're expecing WEIGHTS to be a maxtrix with a column of input weights
       for each neuron in this layer.  */
    assert (weights.shape.size () == 2
	    && weights.shape[1] == v_membrane.size ());
    /* Similarly, we expect one BIAS value per neuron.  */
    assert (bias.size () == v_membrane.size ());

    std::vector<uint32_t> spikes_out;
    for (uint32_t i = 0; i < weights.shape[1]; i++)
      {
	float update = bias[i];
	for (uint32_t j = 0; j < spikes_in.size (); j++)
	  {
	    rts_checking_assert (spikes_in[j] < weights.shape[0]);
	    update += weights.vec[i + (spikes_in[j] * weights.stride[0])];
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

  float beta;
  float v_thresh;
  std::vector<float> v_membrane;
};

#endif // LIF_H_

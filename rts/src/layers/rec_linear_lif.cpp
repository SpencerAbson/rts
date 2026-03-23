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
rec_linear_lif<T>::poll_spiking_output (std::vector<uint32_t> &spikes_out,
					uint32_t batch_begin, uint32_t batch_end)
{
  /* Push the spiking neurons to SPIKES_OUT.  */
  for (uint32_t i = batch_begin; i < batch_end; i++)
    {
      if (this->m_v_membrane[i] > this->m_v_thresh)
	{
	  spikes_out.push_back (i);
	  /* Handle the recurrent weight.  */
	  this->m_v_membrane[i] += m_weights_rec[i];
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

#include <arm_neon.h>
#include <type_traits>
#include "../../include/util.h"
#include "../../include/tensor.hpp"
#include "../../include/layers/full_rec_linear_lif.hpp"

/* NOTE: Dervied members are not automatically brought into scope by
   C++'s templated inheritance rules, hence 'this->'.  */

template<typename T>
full_rec_linear_lif<T>::full_rec_linear_lif
		  (tensor<T> weights, std::vector<T> bias, tensor<T> w_rec,
		   uint32_t batch_size, T beta, T v_thresh, uint64_t batch_cost)
  : linear_lif<T> (weights, bias, batch_size, beta, v_thresh, batch_cost,
		   "FULL_REC_LLIF"),
    m_weights_rec (w_rec),
    /* By default, we assume that there is only one thread reading and writing
       the recurrent spikes.  If the network changes this it must call
       register_num_sublayers.  */
    m_buffer_rec (1, 1)
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for full_rec_linear_lif");

  assert (w_rec.shape.size () == 2 && w_rec.shape[0] == w_rec.shape[1]);
  assert (w_rec.shape[0] == weights.shape[1]);
}

template<typename T>
std::vector<uint32_t>
full_rec_linear_lif<T>::timestep_batched (const std::vector<uint32_t> &spikes_in,
					  uint32_t batch_begin, uint32_t batch_end)
{
  rts_checking_assert (batch_begin < batch_end);

  if constexpr (std::is_same_v<T, float>)
    /* float32_t implementaion.  */
    {
      this->f32_neuron_update (batch_begin, batch_end);
      /* Forward contribution.  */
      this->f32_spike_prop (spikes_in, this->m_weights, batch_begin, batch_end);
      /* Recurrent contribution.  */
      this->f32_spike_prop (m_buffer_rec.read (), m_weights_rec, batch_begin,
			    batch_end);
    }
  else if constexpr (std::is_same_v<T, float16_t>)
    /* float16_t implementation.  */
    {
      this->f16_neuron_update (batch_begin, batch_end);
      /* Forward contribution.  */
      this->f16_spike_prop (spikes_in, this->m_weights, batch_begin, batch_end);
      /* Recurrent contribution.  */
      this->f16_spike_prop (m_buffer_rec.read (), m_weights_rec, batch_begin,
			    batch_end);
    }
  else
    rts_unreachable ("Type construction for full_rec_linear_lif");

  std::vector<uint32_t> spike_out;
  /* Push the spiking neurons to SPIKE_OUT.  */
  for (uint32_t i = batch_begin; i < batch_end; i++)
    {
      if (this->m_v_membrane[i] > this->m_v_thresh)
	spike_out.push_back (i);
    }
  /* Write SPIKE_OUT to the recurrent buffer.  */
  m_buffer_rec.write (spike_out);

  return spike_out;
}

template<typename T>
void
full_rec_linear_lif<T>::register_num_sublayers (uint32_t count)
{
  m_buffer_rec.set_readers (count);
  m_buffer_rec.set_writers (count);
}

template<typename T>
void
full_rec_linear_lif<T>::reset ()
{
  /* Reset membrane potentials to zero.  */
  std::fill (this->m_v_membrane.begin (), this->m_v_membrane.end (), (T)0);
  /* Reset the recurrent spike buffer.  */
  m_buffer_rec.reset ();
}

template<typename T>
void
full_rec_linear_lif<T>::profile_worstcase_batch ()
{
  timespec start, end;
  std::vector<uint32_t> spikes_rec;
  std::vector<uint32_t> input = this->worstcase_input ();

  /* We must also consider recurrent spikes here, so fill the
     recurrent spike buffer with a spike from each neuron.  */
  for (uint32_t i = 0; i < this->m_num_outputs; i++)
    spikes_rec.push_back (i);
  m_buffer_rec.write (spikes_rec);

  /* Measure the execution time.  */
  clock_gettime (CLOCK_MONOTONIC, &start);
  this->timestep_batched (input, 0, this->m_batch_size);
  clock_gettime (CLOCK_MONOTONIC, &end);

  this->m_batch_cost_ns = (end.tv_sec - start.tv_sec) * 1E9
    + end.tv_nsec - start.tv_nsec;

  /* Reset state.  */
  this->reset ();
}

template class full_rec_linear_lif<float32_t>;
template class full_rec_linear_lif<float16_t>;

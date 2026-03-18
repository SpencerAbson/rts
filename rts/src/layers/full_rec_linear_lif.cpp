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
       the recurrent spikes.  */
    m_buffer_rec (this->m_num_outputs, 1, 1)
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for full_rec_linear_lif");

  assert (w_rec.shape.size () == 2 && w_rec.shape[0] == w_rec.shape[1]);
  assert (w_rec.shape[0] == weights.shape[1]);
}

template<typename T>
full_rec_linear_lif<T>::full_rec_linear_lif
		  (std::string path_w, std::string path_b, std::string path_rw,
		   uint32_t num_inputs, uint32_t num_outputs,
		   uint32_t batch_size, T beta, T v_thresh, uint64_t batch_cost)
  : linear_lif<T> (path_w, path_b, num_inputs, num_outputs, batch_size, beta,
		   v_thresh, batch_cost, "FULL_REC_LLIF"),
    m_weights_rec (path_rw, {num_outputs, num_outputs}),
    m_buffer_rec (num_outputs, 1, 1)
{
  static_assert (std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
		 "Invalid type construction for full_rec_linear_lif");
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
full_rec_linear_lif<T>::timestep_batched (const std::vector<uint32_t> &spikes_in,
					  uint32_t batch_begin, uint32_t batch_end)
{
  rts_checking_assert (batch_begin < batch_end);

  /* Update the neurons w.r.t the LIF update rule.  */
  this->neuron_update (batch_begin, batch_end);
  /* Handle the linear spiking input.  */
  this->spike_prop (spikes_in, this->m_weights, batch_begin, batch_end);

  /* Handle the recurrent spiking input.  */
  const std::vector<uint32_t> *rec_spikes = m_buffer_rec.acquire_read ();
  this->spike_prop (*rec_spikes, this->m_weights_rec, batch_begin,
		    batch_end);
  m_buffer_rec.release_read ();
}

template<typename T>
void
full_rec_linear_lif<T>::poll_spiking_output (std::vector<uint32_t> &spikes_out,
					     uint32_t batch_begin,
					     uint32_t batch_end)
{
  std::vector<uint32_t> *ptr = m_buffer_rec.acquire_write ();
  for (uint32_t i = batch_begin; i < batch_end; i++)
    {
      if (this->m_v_membrane[i] > this->m_v_thresh)
	{
	      spikes_out.push_back (i);
	      ptr->push_back (i);
	}
    }
  m_buffer_rec.release_write ();
}

template<typename T>
std::string
full_rec_linear_lif<T>::str_buffers (uint32_t level) const
{
  std::string read = this->m_buffer_rd == nullptr ? ""
    : std::format ("(buff:RD {}) ", this->m_buffer_rd->debug_id ());

  std::string write = this->m_buffer_rd == nullptr ? ""
    : std::format ("(buff:WR {}) ", this->m_buffer_wr->debug_id ());

  return std::format ("{}{}{}(buff:REC {})", std::string (level, '\t'),
		      read, write, m_buffer_rec.debug_id ());
}

template<typename T>
void
full_rec_linear_lif<T>::set_buffer_wr (spikebuffer *buff)
{
  /* The number of writers to this buffer is the number of sublayers
     the network has split this layer into.  We need to forward this
     information to M_BUFFER_REC.  */
  m_buffer_rec.set_readers (buff->writers ());
  m_buffer_rec.set_writers (buff->writers ());

  this->m_buffer_wr = buff;
}

template<typename T>
void
full_rec_linear_lif<T>::profile_worstcase_batch ()
{
  timespec start, end;
  std::vector<uint32_t>* spikes_rec;
  std::vector<uint32_t> input = this->worstcase_input ();

  /* We must also consider recurrent spikes here, so fill the
     recurrent spike buffer with a spike from each neuron.  */
  spikes_rec = m_buffer_rec.acquire_write ();
  for (uint32_t i = 0; i < this->m_num_outputs; i++)
    spikes_rec->push_back (i);

  m_buffer_rec.release_write ();

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

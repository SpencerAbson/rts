#ifndef FULL_REC_LINEAR_LIF_H_
#define FULL_REC_LINEAR_LIF_H_

#include "linear_lif.hpp"
#include "../buffers.hpp"

/* A fully-connected layer of LIF neurons, with a full self connection.  */
template <typename T>
class full_rec_linear_lif : public linear_lif<T>
{
public:
  full_rec_linear_lif (tensor<T> weights, std::vector<T> bias, tensor<T> w_rec,
		       uint32_t batch_size, T beta=(T)0.8, T v_thresh=(T)1.0,
		       uint64_t batch_cost=COST_UNDEF);

  std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t> &spikes_in,
		    uint32_t batch_begin, uint32_t batch_end);

  void
  register_num_sublayers (uint32_t count) override;

  void
  reset ();

  void
  profile_worstcase_batch () override;

private:
  tensor<T> m_weights_rec;
  spikebuffer m_buffer_rec;
};


#endif // FULL_REC_LINEAR_LIF_H_

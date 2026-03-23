#ifndef REC_LINEAR_LIF_H_
#define REC_LINEAR_LIF_H_

#include "linear_lif.hpp"

/* A fully-connected layer of LIF neurons, each with a one-to-one recurrent
   connection.  */
template <typename T>
class rec_linear_lif : public linear_lif<T>
{
public:
  rec_linear_lif (tensor<T> weights, std::vector<T> bias, std::vector<T> w_rec,
		  uint32_t batch_size, T beta=(T)0.8, T v_thresh=(T)1.0,
		  uint64_t batch_cost=COST_UNDEF);

  rec_linear_lif (const std::string &path_w, const std::string &path_b,
		  const std::string &path_rw, uint32_t num_inputs,
		  uint32_t num_outputs, uint32_t batch_size, T beta=(T)0.8,
		  T v_thresh=(T)1.0, uint64_t batch_cost=COST_UNDEF);

  void
  profile_worstcase_batch () override;

  /* We'll modify this to account for the recurrent weighting.  */
  void
  poll_spiking_output (std::vector<uint32_t> &spikes_out,
		       uint32_t batch_begin, uint32_t batch_end) override;

protected:
  std::vector<T> m_weights_rec;
};

#endif // REC_LINEAR_LIF_H_

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

  void
  profile_worstcase_batch () override;

protected:
  /* Update the LIF update rule to each neuron, subtracting the
     recurrent weight if we spiked at the last timestep.  */
  void
  f32_neuron_update (uint32_t batch_begin, uint32_t batch_end) override;
  void
  f16_neuron_update (uint32_t batch_begin, uint32_t batch_end) override;

  std::vector<T> m_weights_rec;
};

#endif // REC_LINEAR_LIF_H_

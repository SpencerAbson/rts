#include <vector>
#include <format>
#include "util.h"
#include "../test_util.h"
#include "layers/full_rec_linear_lif.hpp"

#define BETA 0.905
#define NTIMESTEPS 620

#define NUM_INPUTS  2048
#define NUM_OUTPUTS 400

#define KENDALL_MIN 0.9
#define INPUT_PATH  PROJECT_ROOT"/full_rec_linear_lif/data/inputs"

int main ()
{
  auto lif = full_rec_linear_lif<ftype> (MODEL_PATH "/w_fc1",
					 MODEL_PATH "/b_fc1",
					 MODEL_PATH "/w_rec1",
					 NUM_INPUTS, NUM_OUTPUTS,
					 NUM_OUTPUTS, BETA);

  /* The number of times each output neuron fired a spike in our
     implementation.  */
  std::vector<uint32_t> frequencies_test (NUM_OUTPUTS, 0);
  /* The number of times each output neuron fired a spike in the SNNtorch
     implemtation.  */
  std::vector<uint32_t> frequencies_gold (NUM_OUTPUTS, 0);

  /* Tally the outputs of both implementations.  */
  for (uint32_t i = 0; i < NTIMESTEPS; i++)
    {
      std::vector<uint32_t> spikes;
      if (weights_from_file (std::format (OUTPUT_PATH"/{}", i), spikes))
	return -1;

      for (uint32_t spike : spikes)
	{
	  assert (spike < NUM_OUTPUTS);
	  frequencies_gold[spike]++;
	}

      if (weights_from_file (std::format (INPUT_PATH"/{}", i), spikes))
      	return -1;

      for (uint32_t spike : lif.forward (spikes))
	{
	  assert (spike < NUM_OUTPUTS);
	  frequencies_test[spike]++;
	}
    }

  /* We require that the maximal firing neurons in both implementations
     are exactly the same.  */
  if (!vector_same_contents_p (maximal_indices (frequencies_test),
			       maximal_indices (frequencies_gold)))
    return -1;

  /* Additionally, the distrbutions should look more-or-less the same.  We'll
     use the Kendall rank correlation coefficient to quantify this.  */
  if (kendall_rank_corr (frequencies_test, frequencies_gold) < KENDALL_MIN)
    return -1;

  return 0;
}

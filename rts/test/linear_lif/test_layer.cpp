#include <vector>
#include <format>
#include <algorithm>
#include <utility>
#include "util.h"
#include "layers/linear_lif.hpp"

#define BETA 0.95
#define NTIMESTEPS 100

#define NUM_INPUTS  2312
#define NUM_OUTPUTS 1000

#define KENDALL_MIN 0.9
#define INPUT_PATH  PROJECT_ROOT"/linear_lif/data/inputs"


int main ()
{
  auto lif = linear_lif<ftype> (MODEL_PATH "/w_fc1", MODEL_PATH "/b_fc1",
				NUM_INPUTS, NUM_OUTPUTS, NUM_OUTPUTS, BETA);

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

  /* We require that any maximal element of FREQUENCIES_GOLD is also
     one of FREQUENCIES_TEST (e.g. we'd make the same classification
     decision).  */
  uint32_t max = 0;
  uint32_t idx_max_gold = 0;
  for (uint32_t i = 0; i < NUM_OUTPUTS; i++)
    {
      if (frequencies_gold[i] > max)
	{
	  max = frequencies_gold[i];
	  idx_max_gold = i;
	}
    }

  for (uint32_t i = 0; i < NUM_OUTPUTS; i++)
    {
      if (frequencies_test[i] > frequencies_test[idx_max_gold])
	return -1;
    }

  /* Additionally, the distrbutions should look more-or-less the same.  We'll
     use the Kendall rank correlation coefficient to quantify this.   */
  uint32_t concordant = 0;
  uint32_t discordant = 0;
  for (uint32_t i = 0; i < NUM_OUTPUTS; i++)
    {
      for (uint32_t j = i + 1; j < NUM_OUTPUTS; j++)
	{
	  if ((frequencies_test[i] <= frequencies_test[j])
	      == (frequencies_gold[i] <= frequencies_gold[j]))
	    concordant++;
	  else
	    discordant++;
	}
    }

  if ((double) (concordant - discordant)
      / ((NUM_OUTPUTS * (NUM_OUTPUTS - 1) / 2))  < KENDALL_MIN)
    return -1;

  return 0;
}

#include <vector>
#include <format>
#include "util.h"
#include "../test_util.h"
#include "network.hpp"
#include "layers/rec_linear_lif.hpp"

#define BETA 0.905
#define THRESH_1 0.3
#define THRESH_2 1.0

#define NTIMESTEPS 22
/* Overly conversavative to avoid this failing due to timing.  */
#define TIMESTEP_US 10000

#define NUM_INPUTS  2450
#define NUM_HIDDEN  400
#define NUM_OUTPUTS 4

#define KENDALL_MIN 0.9
#define INPUT_PATH  PROJECT_ROOT"/rec_linear_lif/data/inputs"

int main ()
{
  network net = network (NTHREADS, TIMESTEP_US);

  auto rlif1
    = std::make_unique<rec_linear_lif<ftype>> (MODEL_PATH "/w_fc1",
					       MODEL_PATH "/b_fc1",
					       MODEL_PATH "/w_rec1",
					       NUM_INPUTS, NUM_HIDDEN,
					       /* batch size.  */
					       NUM_HIDDEN / 4, BETA, THRESH_1);
  auto rlif2
    = std::make_unique<rec_linear_lif<ftype>> (MODEL_PATH "/w_fc2",
					       MODEL_PATH "/b_fc2",
					       MODEL_PATH "/w_rec2",
					       NUM_HIDDEN, NUM_OUTPUTS,
					       /* batch size.  */
					       NUM_OUTPUTS, BETA, THRESH_2);
  net.add_layer (std::move (rlif1));
  net.add_layer (std::move (rlif2));

  /* The number of times each output neuron fired a spike in our
     implementation.  */
  std::vector<uint32_t> frequencies_test (NUM_OUTPUTS, 0);
  /* The number of times each output neuron fired a spike in the SNNtorch
     implemtation.  */
  std::vector<uint32_t> frequencies_gold (NUM_OUTPUTS, 0);
  /* Tally the golden output spikes.  */
  for (uint32_t i = 0; i < NTIMESTEPS; i++)
    {
      std::vector<uint32_t> spikes_out;
      if (weights_from_file (std::format (OUTPUT_PATH"/{}", i), spikes_out))
	return -1;

      for (uint32_t spike : spikes_out)
	{
	  assert (spike < NUM_OUTPUTS);
	  frequencies_gold[spike]++;
	}
    }

  auto input = [&net] (bool *killswitch)
  {
    static int ts_count = 0;

    std::vector<uint32_t> spikes_in;
    if (ts_count == (NTIMESTEPS - 1))
      /* Kill the simulation.  */
      *killswitch = true;
    else
      {
	int res
	  = weights_from_file (std::format (INPUT_PATH"/{}", ts_count),
			       spikes_in);
	assert (!res);
      }

    ts_count++;
    return spikes_in;
  };

  auto output = [&frequencies_test]
    (const std::vector<uint32_t> &spikes_out)
  {
    for (uint32_t spike : spikes_out)
      {
	assert (spike < NUM_OUTPUTS);
	frequencies_test[spike]++;
      }
  };

  net.initialise (input, output);
  if (net.run ())
    return -1;

  /* We require that the maximal firing neurons in both implementations
     is exactly the same.  */
  if (!vector_same_contents_p (maximal_indices (frequencies_test),
			       maximal_indices (frequencies_gold)))
    return -1;

  /* Additionally, the distrbutions should look more-or-less the same.  We'll
     use the Kendall rank correlation coefficient to quantify this.  */
  if (kendall_rank_corr (frequencies_test, frequencies_gold) < KENDALL_MIN)
    return -1;

  return 0;

}

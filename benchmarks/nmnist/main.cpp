#include <memory>
#include <vector>
#include <iostream>
#include "util.h"
#include "network.hpp"
#include "layers/linear_lif.hpp"

#define NTIMESTEPS  10
#define NTHREADS    1
#define TIMESTEP_US 1000
#define RATE_MHZ    5E-4

#define BETA 0.95
/* 34 * 34 * 2.  */
#define NUM_INPUTS  2312
#define NUM_HIDDEN  800
#define NUM_OUTPUTS 10
#define MODEL_PATH PROJECT_ROOT"/models/fcsnn-800-f32"


int main ()
{
  network net = network (NTHREADS, TIMESTEP_US);

  /* Create layers.  */
  auto lif1
    = std::make_unique<linear_lif<float>> (MODEL_PATH "/w_fc1",
					   MODEL_PATH "/b_fc1",
					   NUM_INPUTS, NUM_HIDDEN,
					   /* batch size.  */
					   NUM_HIDDEN / 2, BETA);
  auto lif2
    = std::make_unique<linear_lif<float>> (MODEL_PATH "/w_fc2",
					   MODEL_PATH "/b_fc2",
					   NUM_HIDDEN, NUM_OUTPUTS,
					   /* batch size.  */
					   NUM_OUTPUTS, BETA);
  /* Add to network (order is important).  */
  net.add_layer (std::move (lif1));
  net.add_layer (std::move (lif2));

  /* Input and ouput functions.  */
  auto input = [&net] (bool *killswitch)
  {
    static int ts_count = 0;

    if (ts_count == NTIMESTEPS)
      /* Kill the simulation.  */
      *killswitch = true;

    ts_count++;
    return net.generate_poisson_input (RATE_MHZ);
  };

  auto output = [] (const std::vector<uint32_t> &spikes_out)
  {
    /* Do nothing with the output.  */
  };

  /* Init and run.  */
  net.initialise (input, output);
  net.run ();

  /* All statistics are written to the RTS_PERF_DIR directory, but some
     basic latency information is available here.  */
  std::cout << net.generate_performance_overview () << std::endl;

  return 0;
}

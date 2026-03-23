#include <memory>
#include <vector>
#include <iostream>
/* For float32_t and float16_t.  */
#include <arm_neon.h>
#include "network.hpp"
#include "layers/rec_linear_lif.hpp"

/* The samples are ~30ms on average.  */
#define NTIMESTEPS  30
#define NTHREADS    1
/* The model was trained on 1ms frame bins.  */
#define TIMESTEP_US 1000
/* Use 1E-3 to have every input neuron fire at every timstep.  */
#define RATE_MHZ    5E-4

#define BETA     0.95
#define THRESH_1 0.3
#define THRESH_2 1.0

/* 35 * 35 * 2.  */
#define NUM_INPUTS  2450
#define NUM_HIDDEN  400
#define NUM_OUTPUTS 4

/* Toggle this on to switch to the FP16 model.  */
#define USE_FP16 0

#if USE_FP16
#define ftype float16_t
#define MODEL_PATH PROJECT_ROOT"/models/rsnn-400-f16"
#else
#define ftype float32_t
#define MODEL_PATH PROJECT_ROOT"/models/rsnn-400-f32"
#endif

/* The FP32 models acheives ~88% acurracy, the FP16 model acheives
   ~84% accuracy.  */

int main ()
{
  network net = network (NTHREADS, TIMESTEP_US);

  /* Create layers.  */
  auto rlif1
    = std::make_unique<rec_linear_lif<ftype>> (MODEL_PATH "/w_fc1",
					       MODEL_PATH "/b_fc1",
					       MODEL_PATH "/w_rec1",
					       NUM_INPUTS, NUM_HIDDEN,
					       /* batch size.  */
					       NUM_HIDDEN / 2, BETA, THRESH_1);
  auto rlif2
    = std::make_unique<rec_linear_lif<ftype>> (MODEL_PATH "/w_fc2",
					       MODEL_PATH "/b_fc2",
					       MODEL_PATH "/w_rec2",
					       NUM_HIDDEN, NUM_OUTPUTS,
					       /* batch size.  */
					       NUM_OUTPUTS, BETA, THRESH_2);

  /* Add to network (order is important).  */
  net.add_layer (std::move (rlif1));
  net.add_layer (std::move (rlif2));

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

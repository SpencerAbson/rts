#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
/* For float32_t and float16_t.  */
#include <arm_neon.h>
#include "util.h"
#include "network.hpp"
#include "layers/linear_lif.hpp"

/* This benchmark is a sandbox for exploring model parallelism.  We can
   create arbitrary models (with random weights/biases), alter simulation
   parameters, and oberve basic performance information.  */

/* Change these to set the simulation timing information.  */
#define NTIMESTEPS  50
#define TIMESTEP_US 10000
/* Change this to set the spiking frequency of each input
   neuron.  */
#define RATE_MHZ 2E-5

/* Change these to set the model shape.  */
#define NUM_INPUTS  800
#define NUM_HIDDEN  5000
#define NUM_OUTPUTS 50

/* Change these to set the minimum number of neurons that the
   partitioning algorithm can reason with for each layer.  */
#define BATCH_SIZE_1 500
#define BATCH_SIZE_2 50

/* Change this to set the number of threads that the patitioning
   algorithm can use.  */
#define NTHREADS 1

/* Toggle this on to switch to FP16.  */
#define USE_FP16 0

#if USE_FP16
#define ftype float16_t
#else
#define ftype float32_t
#endif

static std::unique_ptr<linear_lif<ftype>>
create_random_layer (uint32_t input_features, uint32_t output_features,
		     uint32_t batch_size)
{
  std::uniform_real_distribution<float> dist (0.0, 1.0);

  tensor<ftype> weights ({input_features, output_features});
  std::vector<ftype> bias (output_features);

  /* Randomise the wights.  */
  std::generate (weights.vec.begin (), weights.vec.end (), [&]()
  {
    return static_cast<ftype> (dist (mersenne_twister ()));
  });
  /* Likewise for the bias.  */
  std::generate (bias.begin (), bias.end (), [&]()
  {
    return static_cast<ftype> (dist (mersenne_twister ()));
  });

  return std::make_unique<linear_lif<ftype>> (weights, bias, batch_size);
}

int main ()
{
  network net = network (NTHREADS, TIMESTEP_US);

  auto lif1 = create_random_layer (NUM_INPUTS, NUM_HIDDEN, BATCH_SIZE_1);
  auto lif2 = create_random_layer (NUM_HIDDEN, NUM_OUTPUTS, BATCH_SIZE_2);

  /* Optionally set our own batch cost, rather than using the profiler.  */
  /* lif1->set_batch_cost (...);  */

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
}

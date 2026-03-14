#include <memory>
#include <vector>
#include <iostream>
#include "util.h"
#include "tensor.hpp"
#include "network.hpp"
#include "layers/linear_lif.hpp"

/* The highest representable firing rate for a
   timestep of 1ms.  */
#define RATE_MHZ    1E-3
#define BETA	    0.95
#define TIMESTEP_US 1000

/* 34 * 34 * 2.  */
#define NUM_INPUTS  2312
#define NUM_HIDDEN  800
#define NUM_OUTPUTS 10
/* Use an mnist model for now.  */
#define MODEL_PATH PROJECT_ROOT"/../nmnist/models/fcsnn-800-f32"


int main ()
{
  /* We won't actually use the former parameter.  */
  network net = network (1, TIMESTEP_US);

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

  net.inference (net.generate_poisson_input (RATE_MHZ));
  return 0;
}

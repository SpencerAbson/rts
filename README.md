# RTS

RTS is a real-time simulation tool for spiking neural networks that was
written to support my undergraduate dissertation at the University of
Manchester.

The purpose of this project is to investigate the potential for
high-performance inference of spiking neural networks on small/edge
systems; it targets the Raspberry Pi-5 (Arm Cortex-A76) only.

The project leverages spike sparsity, vectorisation, network-parallelisation,
a real-time capable kernel (Linux with `PREEMPT_RT`), and the highly efficient
LIF neuron model to acheive fast and deterministic spiking inference.  At the
time of writing, it supports fully-connected layers of LIF neurons, with either
no explicit recurrent connections, one-to-one recurrent connections, or
all-to-all recurrent connections.  Each layer type supports both FP16 and FP32
data.

The simulator runs under the assumption that every model includes a global 1
time step synaptic delay.

Please consider that this is an academic experiment and __not production
software__.

## Building
As mentioned above, the sole target of this work is the Raspberry Pi-5 (Arm
Cortex A76), and indeed the CMakeLists.txt currently contains

`target_compile_options(rts PUBLIC .... -mcpu=cortex-a76)`

.  That said, the project will compile with

`target_compile_options(rts PUBLIC .... -march=armv8.2+fp16`

or any superseding architecture.  All development was done using GCC 14.2,
and GCC 11 or greater is required.


Taking a look at the `benchmarks/nmnist` directory, we can see an example of
how to include this project as a library in your own via CMake
```
cmake_minimum_required(VERSION 3.20.0)
project(nmnist_bench)

# --------------------------------------------------------------
# Definitions
# --------------------------------------------------------------
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
		-DRTS_PERF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/perf"
		-DRTS_EN_PROFILE_NETWORK=1   # Record thread latency/wakeup times
		-DRTS_EN_DEBUG_PRINT=1       # Print debugging info to stderr
		-DRTS_EN_CHECKING_ASSERT=1)  # Sanity assertions

# --------------------------------------------------------------
# Libraries
# --------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../rts rts_build)

add_executable(bench main.cpp)
target_link_libraries(bench PUBLIC rts)
```

Noting that a definition of `RTS_EN_PROFILE_NETWORK` also requires a
defintition of `RTS_PERF_DIR`, which is where the library will write the
profiler information.

This example does not define `RTS_EN_RT_POLICY`, which would tell the libary
to use the [SCHED_FIFO](https://man7.org/linux/man-pages/man7/sched.7.html)
real-time policy when executing the network.  To use this feature, one would
need to build and install a kernel with `PREEMPT_RT` applied.

Building any of the benchmarks should as simple as running `make`.
## Examples

Please see the `/benchmarks` directory for a set of examples.  In general, we
begin by defining a network

```
#include "network.hpp"
#include "layers/linear_lif.hpp"
...
#define NTHREADS    1
#define TIMESTEP_US 1000
...
int main ()
{
  network net = network (NTHREADS, TIMESTEP_US);
  ...
```
before adding sequential layers,

```
  ...
  /* Create layers.  */
  auto lif1 = std::make_unique<linear_lif<float>> (...);
  auto lif2 = std::make_unique<linear_lif<float>> (...);
  /* Add to network.  */
  net.add_layer (std::move (lif1));
  net.add_layer (std::move (lif2));
  ...
```
defining the functions that supply input spikes and process output spikes,
```
  ...
#define NTIMESTEPS  10
#define RATE_MHZ    5E-4
  ...
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
    /* Do some interesting processing.  */
  };
  ...
```
initialising the network,

```
  ...
  net.initialise (input, output);
  ...
```
and finally calling `net.run ()` to begin the simulation.
## Code Format
The entire project tries to follow the standard [GNU formatting guidlines](https://www.gnu.org/prep/standards/standards.html#Formatting)
for C/C++ code.

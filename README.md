
# RTS

RTS is a real-time simulation tool for spiking neural networks that was
written to support my undergraduate dissertation at the University of
Manchester.

The purpose of this project is to investigate the potential for high-performance
inference of spiking neural networks on small/edge  systems; it targets the
Raspberry Pi-5 (Arm Cortex-A76) only.

The project leverages spike sparsity, vectorisation, network-parallelisation,
a real-time capable kernel (Linux with `PREEMPT_RT`), and the highly efficient
LIF neuron model to acheive fast and deterministic spiking inference.

Verification and evaluation was performed against models trained using
[SNNtorch](https://github.com/jeshraghian/snntorch), hence the APIs for each
layer type closely mirror that which is used there. Most importantly:

* `linear_lif` is equivalent to passing the output of an `torch.nn.Linear` to
the input of a `snn.Leaky`.
* `rec_linear_lif` is equivalent to passing the output of an `torch.nn.Linear`
to the input of a `snn.RLeaky` with `all_to_all=False` and an indepenent
recurrent weight per neuron.
* `full_rec_linear_lif` is equivalent to passing the output of an
`torch.nn.Linear` to the input of a `snn.RLeaky` with `all_to_all=True`.  Note
that this class takes only one bias parameter because we may combine those of
the linear and recurrent steps into a single term.

Each layer type supports both IEEE FP32 and FP16 data.

## Simualtor Notes

One of the goals of this project is to explore model parallelism for spiking
networks.  Models can be parallelised arbitrarily; we might have multiple layers
being worked on by the same thread, or multiple threads working on the same
layer.  The simulator attempts to balance the load of each thread using a greedy
algorithm, you can explore this a bit in `benchmarks/model-parallelism`.

To enable this, __a global synaptic delay of one timestep is required__.  Models
trained to run using RTS must take this into accout.

Unlike SNNtorch, spiking input/output is not represented using matrices of 0s
and 1s but in a coordinate format, where a flat coordinate is stored only if
its position in the matrix is nonzero.  This __must be considered__ when
providing input or receiving output.

Additionally, while the weights of a `torch.nn.Linear (in_features, out_features)`
have shape `[out_features, in_features]`, each layer type in RTS assumes they have
shape `[in_features, out_features]` for the sake of performance - consider how
poor the access pattern to a row-major matrix of `[out_features, in_features]`
would be if the input spikes were sparse and spread out.  The effect of this is
that __weights from pytorch must be transposed before use__.

## Building

### Testsuite

To build and run the testsuite:
```bash
cd rts
mkdir build
cmake -S . -B build
cd build
make
ctest
```

### Benchmarks

To build and run any of the benchmarks:
```bash
cd <benchmark>
mkdir build
cmake -S . -B build
cd build
make
./bench
```

### Directives

* Defining `RTS_EN_CHECKING_ASSERT` will enable sanity/checking assertions.
* Defining `RTS_EN_DEBUG_PRINT` will enable debug printing to stderr.
* Defining `RTS_EN_PROFILE_NETWORK` will enable the collection of latency and
wakeup-time statistics during the simulation, which will be written to the
directory pointed to by `RTS_PERF_DIR`; a defintion of `RTS_EN_PROFILE_NETWORK`
__requires__ a definiton of `RTS_PERF_DIR`.
* Defining `RTS_EN_RT_POLICY` will tell the libary to use the
[SCHED_FIFO](https://man7.org/linux/man-pages/man7/sched.7.html) real-time
policy when executing the network.  To use this feature, one would need to build
and install a kernel with `PREEMPT_RT` applied.
* Defining `RTS_EN_LOCK_MEM` will enable a call to `mlockall` before the
simulation is started (see [this commit](https://github.com/SpencerAbson/rts/commit/74c0aad0a81dd3338e65ac7cdda37f391370cea1)).


### Notes

As mentioned above, the sole target of this work is the Raspberry Pi-5
(Arm Cortex A76), and indeed the root CMakeLists.txt currently contains

`target_compile_options(rts PUBLIC .... -mcpu=cortex-a76)`

.  That said, the project will compile with

`target_compile_options(rts PUBLIC .... -march=armv8.2-a+fp16`

or any superseding architecture.  All development was done using GCC 14.2, and
GCC 11 or greater is required.


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
## Examples

In general, we begin by defining a network with timestep and thread-count
parameters.
```cpp
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
The former will depend on how the model in question was trained.  The latter is
configurable and determines how many threads the simulator will split the
execution of the layers into, __this does not include the (2) threads required
to fetch input and process output__.  Given a network, we then add the layers
of a sequential SNN model.
```cpp
  ...
  /* Create layers.  */
  auto lif1 = std::make_unique<linear_lif<float>> (...);
  auto lif2 = std::make_unique<linear_lif<float>> (...);
  /* Add to network.  */
  net.add_layer (std::move (lif1));
  net.add_layer (std::move (lif2));
  ...
```
Before defining the functions to be run by those input/output threads alluded
to above.
```cpp
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
Finally, we can initalise and run the network.
```cpp
  ...
  net.initialise (input, output);
  net.run ();
  ...
```
The network will continue to run until either
* A timing-violation (where a thread has not met it's timing requirement) occurs.
* The simulation is killed by writing to `killswitch` in the input thread.

In both cases the process will clean up and excit gracefully.  Please see the
`benchmarks` directory for a set of examples.

## Schematics

Stringified schematics can be useful for looking at how the simulator has broken
up a network, here is quick reference for parsing them.

```lisp
(network [
	(thread:NET 0 [                       # Thread 0 runs all of...
		(sublayer:LLIF 0 (range 0 800)    # Neurons [0, 800) of layer 0.
			(buff:RD 0) (buff:WR 1))      # Writes to buff 1, reads from 0.
		(sublayer:LLIF 1 (range 0 10)     # Neurons [0, 10) of layer 1.
			(buff:RD 1) (buff:WR 2))      # Writes to buff 2, reads from 1.
	])
	(thread:INP 1 (buff:WR 0))            # Thread 1 runs the input.
	(thread:OUP 2 (buff:RD 2))            # Thread 2 runs the output.
])
```

## Code Format

The entire project tries to follow the standard [GNU formatting guidlines](https://www.gnu.org/prep/standards/standards.html#Formatting) for C/C++ code.

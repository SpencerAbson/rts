#ifndef  NETWORK_H
#define  NETWORK_H

#include <functional>
#include "threads.hpp"
#include "layers/layer.hpp"

class network
{
public:
  network (uint32_t threads=1, uint32_t period_us=1000);
  ~network ();

  /* Add a layer (in sequence) to the network.  */
  void
  add_layer (std::unique_ptr<layer> l);
  /* Distribute the workload, set input/output callbacks, etc.  */
  void
  initialise (input_thread::callback_type input_fn,
	      output_thread::callback_type output_fn);
  /* Run an initialised network.  */
  int
  run ();

  /* The logical (layered) view of the network.  */
  std::string
  str_logical_descr (uint32_t level=0) const;
  /* The schematic (thread-based) view of the initalised network.  */
  std::string
  str_schematic_descr (uint32_t level=0) const;

  /* Model each input neuron as a poisson source using a bernoulli
     approximation.  */
  std::vector<uint32_t>
  generate_poisson_input (double rate_mhz) const;

  /* Do a 'straight-line' single-threaded inference run (no implicit synaptic
     delay) on SPIKES, and return those fired by the last layer.

     This is primarily intended for benchmarking use under EN_PROFILE_NETWORK,
     but it also opens the project up for more general use.  */
  std::vector<uint32_t>
  inference (std::vector<uint32_t> spikes);

private:
  /* Kill all spawned threads.  */
  void
  kill ();
  /* The greedy partiting algorithm used to assign work to threads.  */
  void
  linear_partitioning ();

  /* The number of threads to distribute the workload across.  */
  uint32_t m_num_threads;
  /* RT cyclic period.  */
  uint32_t m_period_us;
  /* The layers of a sequential spiking neural network (in order).  */
  std::vector<std::unique_ptr<layer>> m_layers;
  /* The threads that run this simulation (incl. input/output threads).  */
  std::vector<std::unique_ptr<thread>> m_threads;

  bool m_initialised = false;
};

#endif //  NETWORK_H

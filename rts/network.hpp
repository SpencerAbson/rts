#ifndef  NETWORK_H
#define  NETWORK_H

#include "layers/layer.hpp"

class network
{
public:
  network (uint32_t threads=2, uint64_t period_ns=1000000);
  ~network ();

  /* Add a layer (in sequence) to the network.  */
  void
  add_layer (std::unique_ptr<layer> l);
  /* Distribute the workload, set input/output callbacks, etc.  */
  void
  initialise (std::vector<uint32_t> (*input_cb) (bool *),
	      void (*output_cb) (const std::vector<uint32_t> &));
  /* Run an initialised network.  */
  int
  run ();

  /* The logical (layered) view of the network.  */
  std::string
  str_logical_descr (uint32_t level=0) const;
  /* The schematic (thread-based) view of the initalised network.  */
  std::string
  str_schematic_descr (uint32_t level=0) const;

private:
  /* Kill all spawned threads.  */
  void
  kill ();
  /* The greedy partiting algorithm used to assign work to threads.  */
  void
  linear_partitioning ();

  /* The number of threads used to parallelise the network.  */
  uint32_t m_num_threads;
  /* RT cyclic period.  */
  uint64_t m_period_ns;
  /* The layers of a sequential spiking neural network (in order).  */
  std::vector<std::unique_ptr<layer>> m_layers;

  /* RT thread objects.  */
  input_rtt *m_input_thread = nullptr;
  std::vector<std::unique_ptr<rt_thread>> m_threads;

  bool m_initialised = false;
};

#endif //  NETWORK_H

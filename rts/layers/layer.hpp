#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include "../buffers.hpp"

#define COST_UNDEF 0

class layer
{
public:
  layer (uint32_t num_inputs, uint32_t num_outputs, uint32_t batch_size,
	 uint64_t batch_cost_ns)
    : m_num_inputs (num_inputs), m_num_outputs (num_outputs),
      m_batch_size (batch_size), m_batch_cost_ns (batch_cost_ns)
  {
    assert (num_outputs % batch_size == 0);
  }

  /* Simulate one timestep of the entire layer.  */
  virtual std::vector<uint32_t> timestep (const std::vector<uint32_t>&) = 0;
  /* Simulate one timestep for a subbatch of this layer.  */
  virtual std::vector<uint32_t> timestep_batched (const std::vector<uint32_t>&,
						  uint32_t, uint32_t) = 0;
  /* Many factors influence performance at runtime.  But in the absence of user-
     provided profile information (COST_UNDEF), a rough estimate can be made by
     profiling each layer at their own worst case input.  */
  virtual void time_worstcase (uint32_t iterations, uint32_t batch_size,
			       struct timespec &tstart,
			       struct timespec &tend) = 0;

  uint32_t input_size ()  const
  {
    return m_num_inputs;
  }

  uint32_t output_size () const
  {
    return m_num_outputs;
  }
  /* A profile based estimate of the cost of running TIMESTEP across an
     entire batch, or COST_UNDEF if we lack this information.  */
  uint64_t batch_cost () const
  {
    return m_batch_cost_ns;
  }

  uint32_t batch_size () const
  {
    return m_batch_size;
  }

  uint32_t total_batches () const
  {
    return m_num_outputs / m_batch_size;
  }

  virtual ~layer () = default;

private:
  friend class network;

  void
  profile_batch (uint32_t iterations)
  {
    struct timespec start, end;
    time_worstcase (iterations, m_batch_size, start, end);

    m_batch_cost_ns = ((end.tv_sec - start.tv_sec) * 1000000000
	+ end.tv_nsec - start.tv_nsec) / iterations;

    debug_printf ("\nProfiler information\niterations: %u\nbatch size: \
%u\ncost: %lu (ns)\n", iterations, m_batch_size, m_batch_cost_ns);
  }

  std::vector<uint32_t>
  read ()
  {
    assert (m_buffer_rd != nullptr);
    return m_buffer_rd->read ();
  }

  void
  write (const std::vector<uint32_t> &spikes)
  {
    assert (m_buffer_wr != nullptr);
    m_buffer_wr->write (spikes);
  }

  /* Simulate one timestep using input spikes read from M_BUFFER_RD, and write
     any output spikes to M_BUFFER_WR.

     NOTE: This process generally involves quite a few std::vector push/copy
     operations.  We ought to reserve this space ahead of time to avoid dynamic
     allocation within the RT critical path.  */
  void
  run (uint32_t begin, uint32_t end)
  {
    write (timestep_batched (read (), begin, end));
  }

  /* Lifetimes managed by the network.  */
  spikebuffer *m_buffer_rd = nullptr;
  spikebuffer *m_buffer_wr = nullptr;

protected:
  /* Generic layer info.  */
  uint32_t m_num_inputs;
  uint32_t m_num_outputs;

  uint32_t m_batch_size;
  uint64_t m_batch_cost_ns;
};

#endif // LAYER_H_

#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include "../util.h"
#include "../buffers.hpp"

#define COST_UNDEF 0

class layer
{
  friend class network;
  static uint32_t m_debug_id_counter;
public:
  layer (uint32_t num_inputs, uint32_t num_outputs, uint32_t batch_size,
	 uint64_t batch_cost_ns, std::string debug_type);

  virtual ~layer () = default;

  /* Simulate a timestep for neurons [batch_begin, batch_end). */
  virtual std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t>&, uint32_t batch_begin,
		    uint32_t batch_end) = 0;

  /* Simulate a timestep of the entire layer (uses the above).  */
  std::vector<uint32_t>
  timestep (const std::vector<uint32_t> &spikes_in);

  /* Simulate a timestep for neurons [batch_begin, batch_end) using input
     spikes read from M_BUFFER_RD, and write any output spikes to M_BUFFER_WR.

     Used during simulation only.  */
  void
  run (uint32_t batch_begin, uint32_t batch_end);

  /* The input that should incurr the highest latency; used for self-profiling
     in the absence of user-provided profile information (COST_UNDEF).  */
  virtual std::vector<uint32_t>
  worstcase_input () = 0;

  /* Logical view.  */
  std::string
  str_descr (uint32_t level=0) const;

  /* Schematic view helper.  */
  virtual std::string
  str_buffers (uint32_t level=0) const;

  /* E.g. 'LIF'/'RLIF' etc.  */
  const std::string&
  debug_type () const { return m_debug_type; }

  /* An identifier that is unique among all layers.  */
  uint32_t
  debug_id () const { return m_debug_id; }

  /* The number of input neurons.  */
  uint32_t
  input_size () const { return m_num_inputs; }

  /* The number of neurons in the layer.  */
  uint32_t
  output_size () const { return m_num_outputs; }

  /* A profile based estimate of the cost of simulating an entire batch, or
     COST_UNDEF if we lack this information.  */
  uint64_t
  batch_cost () const { return m_batch_cost_ns; }

  /* The fixed size of every batch.  */
  uint32_t
  batch_size () const { return m_batch_size; }

  uint32_t
  total_batches () const { return m_num_outputs / m_batch_size; }

  virtual void
  set_buffer_rd (spikebuffer *buff) { m_buffer_rd = buff; }

  virtual void
  set_buffer_wr (spikebuffer *buff) { m_buffer_wr = buff; }

  /* Reset the state of any variable dynamics.  */
  virtual void
  reset () = 0;

private:
  std::vector<uint32_t>
  read ();

  void
  write (const std::vector<uint32_t> &spikes);

protected:
  /* Lifetimes managed by the network.  */
  spikebuffer *m_buffer_rd = nullptr;
  spikebuffer *m_buffer_wr = nullptr;

  /* An estimate of the worst-case latency.  Virtual as some layers (recurrent)
     may have dynamics which affect their latency other than just the input.  */
  virtual void
  profile_worstcase_batch ();

  /* Generic layer info.  */
  uint32_t m_num_inputs;
  uint32_t m_num_outputs;

  uint32_t m_batch_size;
  uint64_t m_batch_cost_ns;

  std::string m_debug_type;
  uint32_t m_debug_id;
};

#endif // LAYER_H_

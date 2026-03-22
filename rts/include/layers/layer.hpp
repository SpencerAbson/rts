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

  /* The input that should incurr the highest latency; used for self-profiling
     in the absence of user-provided profile information (COST_UNDEF).  */
  virtual std::vector<uint32_t>
  worstcase_input () = 0;

  /* Reset the state of any dynamic data.  */
  virtual void
  reset () = 0;

  /* Run the entire layer for a single timestep, returning the spiking
     output.  This is for serieal inference only, i.e. it is not suitable
     for use on the RT critical-path.  */
  std::vector<uint32_t>
  forward (const std::vector<uint32_t> &spikes_in);

  /* Run the sublayer [batch_begin, batch_end), reading input from
     M_BUFFER_RD and writing the spiking output to M_BUFFER_WR.  */
  void
  run (uint32_t batch_begin, uint32_t batch_end);

  /* The number of input neurons.  */
  uint32_t
  input_size () const;
  /* The number of neurons in the layer.  */
  uint32_t
  output_size () const;

  /* A profile based estimate of the cost of simulating an entire batch, or
     COST_UNDEF if we lack this information.  */
  uint64_t
  batch_cost () const;
  /* The fixed size of every batch.  */
  uint32_t
  batch_size () const;

  /* Logical view.  */
  std::string
  str_descr (uint32_t level=0) const;
  /* Schematic view helper.  */
  virtual std::string
  str_buffers (uint32_t level=0) const;

  /* An identifier that is unique among all layers.  */
  uint32_t
  debug_id () const;
  /* E.g. 'LIF'/'RLIF' etc.  */
  const std::string&
  debug_type () const;

protected:
  virtual void
  set_buffer_rd (spikebuffer *buff);
  virtual void
  set_buffer_wr (spikebuffer *buff);

  /* Simulate a timestep for neurons [batch_begin, batch_end). */
  virtual void
  timestep_batched (const std::vector<uint32_t>&, uint32_t batch_begin,
		    uint32_t batch_end) = 0;

  virtual void
  poll_spiking_output (std::vector<uint32_t>& out, uint32_t batch_begin,
		       uint32_t batch_end) = 0;

  /* An estimate of the worst-case latency.  Virtual as some layers (recurrent)
     may have dynamics which affect their latency other than just the input.  */
  virtual void
  profile_worstcase_batch ();

  /* Lifetimes managed by the network.  */
  spikebuffer *m_buffer_rd = nullptr;
  spikebuffer *m_buffer_wr = nullptr;

  /* Generic layer info.  */
  uint32_t m_num_inputs;
  uint32_t m_num_outputs;

  uint32_t m_batch_size;
  uint64_t m_batch_cost_ns;

  std::string m_debug_type;
  uint32_t m_debug_id;
};

#endif // LAYER_H_

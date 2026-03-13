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

  /* Simulate one timestep for a subbatch of this layer.  */
  virtual std::vector<uint32_t>
  timestep_batched (const std::vector<uint32_t>&, uint32_t, uint32_t) = 0;
  /* Many factors influence performance at runtime.  But in the absence of user-
     provided profile information (COST_UNDEF), a rough estimate can be made by
     profiling each layer at their own worst case input (see
     profile_worstcase_batch).  */
  virtual std::vector<uint32_t>
  worstcase_input () = 0;

  std::string
  str_descr (uint32_t level=0) const;

  const std::string&
  debug_type () const
  {
    return m_debug_type;
  }

  uint32_t
  debug_id () const
  {
    return m_debug_id;
  }

  uint32_t
  buffer_rd_debug_id () const
  {
    rts_checking_assert (m_buffer_rd != nullptr);
    return m_buffer_rd->debug_id ();
  }

  uint32_t
  buffer_wr_debug_id () const
  {
    rts_checking_assert (m_buffer_wr != nullptr);
    return m_buffer_wr->debug_id ();
  }

  uint32_t
  input_size ()  const
  {
    return m_num_inputs;
  }

  uint32_t
  output_size () const
  {
    return m_num_outputs;
  }
  /* A profile based estimate of the cost of running TIMESTEP across an
     entire batch, or COST_UNDEF if we lack this information.  */
  uint64_t
  batch_cost () const
  {
    return m_batch_cost_ns;
  }

  uint32_t
  batch_size () const
  {
    return m_batch_size;
  }

  uint32_t
  total_batches () const
  {
    return m_num_outputs / m_batch_size;
  }

  /* Reset the state of any variable dynamics.  */
  virtual void
  reset () = 0;
  /* Simulate one timestep of the entire layer.  */
  std::vector<uint32_t>
  timestep (const std::vector<uint32_t> &spikes_in);

  /* Simulate one timestep using input spikes read from M_BUFFER_RD, and write
     any output spikes to M_BUFFER_WR.

     NOTE: This process generally involves quite a few std::vector push/copy
     operations.  We ought to reserve this space ahead of time to avoid dynamic
     allocation within the RT critical path.  */
  void
  run (uint32_t batch_begin, uint32_t batch_end);

private:
  std::vector<uint32_t>
  read ();

  void
  write (const std::vector<uint32_t> &spikes);

  /* Lifetimes managed by the network.  */
  spikebuffer *m_buffer_rd = nullptr;
  spikebuffer *m_buffer_wr = nullptr;

protected:
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

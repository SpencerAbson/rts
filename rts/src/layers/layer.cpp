#include <cassert>
#include "../../include/util.h"
#include "../../include/layers/layer.hpp"

uint32_t layer::m_debug_id_counter = 0;

layer::layer (uint32_t num_inputs, uint32_t num_outputs, uint32_t batch_size,
	      uint64_t batch_cost_ns, std::string debug_type)
  : m_num_inputs (num_inputs), m_num_outputs (num_outputs),
    m_batch_size (batch_size), m_batch_cost_ns (batch_cost_ns),
    m_debug_type (debug_type), m_debug_id (m_debug_id_counter)
{
  assert (num_outputs % batch_size == 0);
  m_debug_id_counter++;
}

std::vector<uint32_t>
layer::forward (const std::vector<uint32_t> &spikes_in)
{
  std::vector<uint32_t> spike_out;
  timestep_batched (spikes_in, 0, m_num_outputs);

  /* Push the spiking neurons to SPIKE_OUT.  */
  poll_spiking_output (spike_out, 0, m_num_outputs);

  return spike_out;
}

std::string
layer::str_descr (uint32_t level) const
{
  return std::format ("{}(layer:{} {} (size {} {}))", std::string (level, '\t'),
		      m_debug_type, m_debug_id, m_num_inputs, m_num_outputs);
}

std::string
layer::str_buffers (uint32_t level) const
{
  std::string read = m_buffer_rd == nullptr ? ""
    : std::format ("(buff:RD {}) ", m_buffer_rd->debug_id ());

  std::string write = m_buffer_rd == nullptr ? ""
    : std::format ("(buff:WR {})", m_buffer_wr->debug_id ());

  return std::format ("{}{}{}", std::string (level, '\t'), read, write);
}

uint32_t
layer::input_size () const
{
  return m_num_inputs;
}

uint32_t
layer::output_size () const
{
  return m_num_outputs;
}

uint64_t
layer::batch_cost () const
{
  return m_batch_cost_ns;
}

uint32_t
layer::batch_size () const
{
  return m_batch_size;
}

uint32_t
layer::debug_id () const
{
  return m_debug_id;
}

const std::string&
layer::debug_type () const
{
  return m_debug_type;
}

void
layer::set_buffer_rd (spikebuffer *buff)
{
  m_buffer_rd = buff;
}

void
layer::set_buffer_wr (spikebuffer *buff)
{
  m_buffer_wr = buff;
}

void
layer::run (uint32_t begin, uint32_t end)
{
  rts_checking_assert (m_buffer_rd && m_buffer_wr);

  const std::vector<uint32_t> *ptr_rd = m_buffer_rd->acquire_read ();
  timestep_batched (*ptr_rd, begin, end);
  m_buffer_rd->release_read ();

  std::vector<uint32_t> *ptr_wr = m_buffer_wr->acquire_write ();
  poll_spiking_output (*ptr_wr, begin, end);
  m_buffer_wr->release_write ();
}

void
layer::profile_worstcase_batch ()
{
  timespec start, end;
  std::vector<uint32_t> input = worstcase_input ();

  /* Measure the execution time under the heaviest load.  */
  clock_gettime (CLOCK_MONOTONIC, &start);
  timestep_batched (input, 0, m_batch_size);
  clock_gettime (CLOCK_MONOTONIC, &end);

  m_batch_cost_ns = (end.tv_sec - start.tv_sec) * 1E9
    + end.tv_nsec - start.tv_nsec;

  /* Reset state.  */
  reset ();
}

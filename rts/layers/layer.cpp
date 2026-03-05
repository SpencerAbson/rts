#include <cassert>
#include "../util.h"
#include "layer.hpp"

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

std::string
layer::str_descr (uint32_t level) const
{
  return std::format ("{}(layer:{} {} (size {} {}))", std::string (level, '\t'),
		      m_debug_type, m_debug_id, m_num_inputs, m_num_outputs);
}

std::vector<uint32_t>
layer::timestep (const std::vector<uint32_t> &spikes_in)
{
  return timestep_batched (spikes_in, 0, m_num_outputs);
}

void
layer::run (uint32_t begin, uint32_t end)
{
  write (timestep_batched (read (), begin, end));
}

void
layer::profile_batch ()
{
  m_batch_cost_ns = time_batch_worstcase_ns ();
  debug_msg ("\nProfiler information\n\nbatch size: \
{}\ncost: {} (ns)\n", m_batch_size, m_batch_cost_ns);
}

std::vector<uint32_t>
layer::read ()
{
  assert (m_buffer_rd != nullptr);
  return m_buffer_rd->read ();
}

void
layer::write (const std::vector<uint32_t> &spikes)
{
  assert (m_buffer_wr != nullptr);
  m_buffer_wr->write (spikes);
}

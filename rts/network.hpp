#ifndef  NETWORK_H
#define  NETWORK_H

#include "layers/layer.hpp"
#include "buffers.hpp"


class network
{
public:
  network (uint32_t threads=2, uint64_t sleep_ns=80000)
    : m_threads (threads), m_sleep_ns (sleep_ns)
  {};

  void
  add_layer (layer *l)
  {
    if (!m_layers.empty ())
      assert (l->input_size () == m_layers.back ()->output_size ()
	      && "Mismatched layer dimensions.");
    m_layers.push_back (l);
  }

  bool
  write (const std::vector<uint32_t> &vec)
  {
    assert (!m_stages.empty () && m_alive);
    return m_stages[0]->buffer_rd->write (vec);
  }

  void
  write_blocking (const std::vector<uint32_t> &vec)
  {
    struct timespec sleep;
    sleep.tv_sec = 0;
    sleep.tv_nsec = m_sleep_ns;

    while (!(write (vec)))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
  }

  bool
  read (std::vector<uint32_t> &vec)
  {
    assert (!m_layers.empty ());
    return (*--m_stages.end ())->buffer_wr->latch (vec);
  }

  void
  read_blocking (std::vector<uint32_t> &vec)
  {
    struct timespec sleep;
    sleep.tv_sec = 0;
    sleep.tv_nsec = m_sleep_ns;

    while (!(read (vec)))
      clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
  }

  void
  kill ()
  {
    m_alive = false;
    for (auto stage : m_stages)
      stage->alive = false;
  }

  bool alive () { return m_alive; }

private:

  static void*
  stage_worker (void *arg)
  {
    stage_info *s_info = (stage_info *)arg;
    /* We don't want to be deferencing these every cycle, so latch
       them here.  */
    buffer<uint32_t> *buffer_rd = s_info->buffer_rd;
    buffer<uint32_t> *buffer_wr = s_info->buffer_wr;
    std::vector<layer *> layers = s_info->layers;

    /* For nanosleeping.  */
    struct timespec sleep;
    sleep.tv_sec  = 0;
    sleep.tv_nsec = s_info->sleep_ns;

    std::vector<uint32_t> result;
    while (s_info->alive)
      {
	/* Wait for input to be ready.  */
	while (!(buffer_rd->latch (result)) && s_info->alive)
	  clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);

	/* Run the layers covered by this stage.  */
	for (auto layer : layers)
	  result = layer->timestep (result);

	/* Wait for output to be latchable?  */
	while (!(buffer_wr->write (result)) && s_info->alive)
	  clock_nanosleep (CLOCK_MONOTONIC, 0, &sleep, NULL);
      }

    return NULL;
  }

  bool m_alive;
  uint32_t m_threads;
  uint64_t m_sleep_ns;

  struct stage_info
  {
    /* layer(s) run during this stage.  */
    std::vector<layer *> layers;
    /* Input buffer for first layer of this stage.  */
    buffer<uint32_t> *buffer_rd;
    /* Output buffer for the last layer this stage.  */
    buffer<uint32_t> *buffer_wr;
    /* Copy of network-wide sleep paramter.  */
    uint64_t sleep_ns;
    /* Killswitch.  */
    bool alive;
  };

  std::vector<layer *> m_layers;
  std::vector<stage_info *> m_stages;
};

#endif //  NETWORK_H

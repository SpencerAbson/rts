/* For float32_t and float16_t.  */
#include <arm_neon.h>

#define MODEL_PATH  PROJECT_ROOT"/linear_lif/models/fcsnn-1000-f32"
#define OUTPUT_PATH PROJECT_ROOT"/linear_lif/data/outputs/net_f32"

#define ftype float32_t

#define NTHREADS 5

#include "test_network.cpp"

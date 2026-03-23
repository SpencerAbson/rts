/* For float32_t and float16_t.  */
#include <arm_neon.h>

#define MODEL_PATH  PROJECT_ROOT"/rec_linear_lif/models/rsnn-400-f16"
#define OUTPUT_PATH PROJECT_ROOT"/rec_linear_lif/data/outputs/net_f16"

#define ftype float16_t

#define NTHREADS 2

#include "test_network.cpp"

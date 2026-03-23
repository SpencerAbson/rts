/* For float32_t and float16_t.  */
#include <arm_neon.h>

#define MODEL_PATH  PROJECT_ROOT"/rec_linear_lif/models/rsnn-400-f32"
#define OUTPUT_PATH PROJECT_ROOT"/rec_linear_lif/data/outputs/layer_f32"

#define ftype float32_t

#include "test_layer.cpp"

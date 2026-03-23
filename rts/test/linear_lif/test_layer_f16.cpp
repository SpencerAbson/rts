/* For float32_t and float16_t.  */
#include <arm_neon.h>

#define MODEL_PATH  PROJECT_ROOT"/linear_lif/models/fcsnn-1000-f16"
#define OUTPUT_PATH PROJECT_ROOT"/linear_lif/data/outputs/layer_f16"

#define ftype float16_t

#include "test_layer.cpp"

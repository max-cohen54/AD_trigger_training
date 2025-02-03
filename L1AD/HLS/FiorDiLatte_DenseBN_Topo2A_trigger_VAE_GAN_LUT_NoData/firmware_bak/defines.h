#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 44
#define N_LAYER_2 32
#define N_LAYER_2 32
#define N_LAYER_5 16
#define N_LAYER_5 16
#define N_LAYER_5 16
#define N_LAYER_8 3
#define N_LAYER_8 3

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<19,11> input_t;
typedef ap_fixed<19,11> dense1_accum_t;
typedef ap_fixed<19,11> layer2_t;
typedef ap_fixed<10,6> weight2_t;
typedef ap_fixed<10,6> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<15,0,AP_RND_CONV,AP_SAT> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<19,11> dense2_accum_t;
typedef ap_fixed<14,6> layer5_t;
typedef ap_fixed<10,6> weight5_t;
typedef ap_fixed<10,6> bias5_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<19,11> layer6_t;
typedef ap_fixed<18,8> dense2_linear_table_t;
typedef ap_ufixed<15,0,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<19,11> z_mean_accum_t;
typedef ap_fixed<12,4> layer8_t;
typedef ap_fixed<10,6> weight8_t;
typedef ap_fixed<10,6> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<19,11> result_t;
typedef ap_fixed<18,8> z_mean_linear_table_t;

#endif

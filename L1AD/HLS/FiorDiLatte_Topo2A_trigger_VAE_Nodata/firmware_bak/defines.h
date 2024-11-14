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
#define N_LAYER_2 32
#define N_LAYER_2 32
#define N_LAYER_6 16
#define N_LAYER_6 16
#define N_LAYER_6 16
#define N_LAYER_6 16
#define N_LAYER_10 3
#define N_LAYER_10 3

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<15,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<38,8> layer2_t;
typedef ap_fixed<10,6> weight2_t;
typedef ap_fixed<10,6> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> dense1_linear_table_t;
typedef ap_fixed<32,7> layer4_t;
typedef ap_fixed<15,6> bn1_scale_t;
typedef ap_fixed<8,0> bn1_bias_t;
typedef ap_ufixed<15,0,AP_RND_CONV,AP_SAT> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<22,3> layer6_t;
typedef ap_fixed<10,6> weight6_t;
typedef ap_fixed<10,6> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> dense2_linear_table_t;
typedef ap_fixed<26,4> layer8_t;
typedef ap_fixed<15,6> bn2_scale_t;
typedef ap_fixed<8,0> bn2_bias_t;
typedef ap_ufixed<15,0,AP_RND_CONV,AP_SAT> layer9_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<23,3> layer10_t;
typedef ap_fixed<10,6> weight10_t;
typedef ap_fixed<10,6> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> z_mean_linear_table_t;

#endif

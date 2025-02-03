#include <iostream>

#include "Topo2A_AD_proj.h"
#include "parameters.h"

void Topo2A_AD_proj(
    input_t inputs[N_INPUT_1_1],
    result_t layer9_out[N_LAYER_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inputs complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=inputs,layer9_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 1408>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 512>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 48>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 3>(b8, "b8.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(inputs, layer2_out, w2, b2); // dense1

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // dense2

    layer6_t layer6_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::linear<layer5_t, layer6_t, linear_config6>(layer5_out, layer6_out); // dense2_linear

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // relu2

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // z_mean

    nnet::linear<layer8_t, result_t, linear_config9>(layer8_out, layer9_out); // z_mean_linear

}

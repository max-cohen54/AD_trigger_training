#include <iostream>

#include "Topo2A_AD_proj.h"
#include "parameters.h"

void Topo2A_AD_proj(
    input_t inputs[N_INPUT_1_1],
    result_t layer11_out[N_LAYER_10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inputs complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=inputs,layer11_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 1408>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<bn1_scale_t, 32>(s4, "s4.txt");
        nnet::load_weights_from_txt<bn1_bias_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 512>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<bn2_scale_t, 16>(s8, "s8.txt");
        nnet::load_weights_from_txt<bn2_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight10_t, 48>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 3>(b10, "b10.txt");
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
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "dense1", N_LAYER_2);
#endif

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // dense1_linear
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "dense1_linear", N_LAYER_2);
#endif

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::normalize<layer3_t, layer4_t, config4>(layer3_out, layer4_out, s4, b4); // BN1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "BN1", N_LAYER_2);
#endif

    layer5_t layer5_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // relu1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer5_t>(layer5_out, "relu1", N_LAYER_2);
#endif

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // dense2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t>(layer6_out, "dense2", N_LAYER_6);
#endif

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::linear<layer6_t, layer7_t, linear_config7>(layer6_out, layer7_out); // dense2_linear
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "dense2_linear", N_LAYER_6);
#endif

    layer8_t layer8_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, layer8_out, s8, b8); // BN2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "BN2", N_LAYER_6);
#endif

    layer9_t layer9_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // relu2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer9_t>(layer9_out, "relu2", N_LAYER_6);
#endif

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // z_mean
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer10_t>(layer10_out, "z_mean", N_LAYER_10);
#endif

    nnet::linear<layer10_t, result_t, linear_config11>(layer10_out, layer11_out); // z_mean_linear
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer11_out, "z_mean_linear", N_LAYER_10);
#endif

}

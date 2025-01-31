#include <iostream>

#include "Topo2A_AD_proj.h"
#include "parameters.h"

typedef ap_fixed<30,21> input_preshift_t;

void Topo2A_AD_proj(
    int in_jet_pt_0,
    int in_jet_eta_0,
    int in_jet_phi_0,
    int in_jet_pt_1,
    int in_jet_eta_1,
    int in_jet_phi_1,
    int in_jet_pt_2,
    int in_jet_eta_2,
    int in_jet_phi_2,
    int in_jet_pt_3,
    int in_jet_eta_3,
    int in_jet_phi_3,
    int in_jet_pt_4,
    int in_jet_eta_4,
    int in_jet_phi_4,
    int in_jet_pt_5,
    int in_jet_eta_5,
    int in_jet_phi_5,
    int in_etau_pt_0,
    int in_etau_eta_0,
    int in_etau_phi_0,
    int in_etau_pt_1,
    int in_etau_eta_1,
    int in_etau_phi_1,
    int in_etau_pt_2,
    int in_etau_eta_2,
    int in_etau_phi_2,
    int in_etau_pt_3,
    int in_etau_eta_3,
    int in_etau_phi_3,
    int in_mu_pt_0,
    int in_mu_eta_0,
    int in_mu_phi_0,
    int in_mu_pt_1,
    int in_mu_eta_1,
    int in_mu_phi_1,
    int in_mu_pt_2,
    int in_mu_eta_2,
    int in_mu_phi_2,
    int in_mu_pt_3,
    int in_mu_eta_3,
    int in_mu_phi_3,
    int in_met_pt_0,
    int in_met_phi_0,
    result_t layer11_out[N_LAYER_10]
) {

    input_preshift_t preshift[N_INPUT_1_1];

    input_t inputs[N_INPUT_1_1];
    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inputs complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=preshift complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_none port=in_jet_pt_0,in_jet_eta_0,in_jet_phi_0,in_jet_pt_1,in_jet_eta_1,in_jet_phi_1,in_jet_pt_2,in_jet_eta_2,in_jet_phi_2,in_jet_pt_3,in_jet_eta_3,in_jet_phi_3,in_jet_pt_4,in_jet_eta_4,in_jet_phi_4,in_jet_pt_5,in_jet_eta_5,in_jet_phi_5,in_etau_pt_0,in_etau_eta_0,in_etau_phi_0,in_etau_pt_1,in_etau_eta_1,in_etau_phi_1,in_etau_pt_2,in_etau_eta_2,in_etau_phi_2,in_etau_pt_3,in_etau_eta_3,in_etau_phi_3,in_mu_pt_0,in_mu_eta_0,in_mu_phi_0,in_mu_pt_1,in_mu_eta_1,in_mu_phi_1,in_mu_pt_2,in_mu_eta_2,in_mu_phi_2,in_mu_pt_3,in_mu_eta_3,in_mu_phi_3,in_met_pt_0,in_met_phi_0,layer11_out 
    #pragma HLS PIPELINE 

    preshift[0] = input_preshift_t(in_jet_pt_0);
    preshift[1] = input_preshift_t(in_jet_eta_0);
    preshift[2] = input_preshift_t(in_jet_phi_0);
    preshift[3] = input_preshift_t(in_jet_pt_1);
    preshift[4] = input_preshift_t(in_jet_eta_1);
    preshift[5] = input_preshift_t(in_jet_phi_1);
    preshift[6] = input_preshift_t(in_jet_pt_2);
    preshift[7] = input_preshift_t(in_jet_eta_2);
    preshift[8] = input_preshift_t(in_jet_phi_2);
    preshift[9] = input_preshift_t(in_jet_pt_3);
    preshift[10] = input_preshift_t(in_jet_eta_3);
    preshift[11] = input_preshift_t(in_jet_phi_3);
    preshift[12] = input_preshift_t(in_jet_pt_4);
    preshift[13] = input_preshift_t(in_jet_eta_4);
    preshift[14] = input_preshift_t(in_jet_phi_4);
    preshift[15] = input_preshift_t(in_jet_pt_5);
    preshift[16] = input_preshift_t(in_jet_eta_5);
    preshift[17] = input_preshift_t(in_jet_phi_5);
    preshift[18] = input_preshift_t(in_etau_pt_0);
    preshift[19] = input_preshift_t(in_etau_eta_0);
    preshift[20] = input_preshift_t(in_etau_phi_0);
    preshift[21] = input_preshift_t(in_etau_pt_1);
    preshift[22] = input_preshift_t(in_etau_eta_1);
    preshift[23] = input_preshift_t(in_etau_phi_1);
    preshift[24] = input_preshift_t(in_etau_pt_2);
    preshift[25] = input_preshift_t(in_etau_eta_2);
    preshift[26] = input_preshift_t(in_etau_phi_2);
    preshift[27] = input_preshift_t(in_etau_pt_3);
    preshift[28] = input_preshift_t(in_etau_eta_3);
    preshift[29] = input_preshift_t(in_etau_phi_3);
    preshift[30] = input_preshift_t(in_mu_pt_0);
    preshift[31] = input_preshift_t(in_mu_eta_0);
    preshift[32] = input_preshift_t(in_mu_phi_0);
    preshift[33] = input_preshift_t(in_mu_pt_1);
    preshift[34] = input_preshift_t(in_mu_eta_1);
    preshift[35] = input_preshift_t(in_mu_phi_1);
    preshift[36] = input_preshift_t(in_mu_pt_2);
    preshift[37] = input_preshift_t(in_mu_eta_2);
    preshift[38] = input_preshift_t(in_mu_phi_2);
    preshift[39] = input_preshift_t(in_mu_pt_3);
    preshift[40] = input_preshift_t(in_mu_eta_3);
    preshift[41] = input_preshift_t(in_mu_phi_3);
    preshift[42] = input_preshift_t(in_met_pt_0);
    preshift[43] = input_preshift_t(in_met_phi_0);

    inputs[0] = preshift[0] >> 8;
    inputs[1] = preshift[1] >> 4;
    inputs[2] = preshift[2] >> 5;
    inputs[3] = preshift[3] >> 7;
    inputs[4] = preshift[4] >> 3;
    inputs[5] = preshift[5] >> 4;
    inputs[6] = preshift[6] >> 6;
    inputs[7] = preshift[7] >> 3;
    inputs[8] = preshift[8] >> 3;
    inputs[9] = preshift[9] >> 5;
    inputs[10] = preshift[10] >> 2;
    inputs[11] = preshift[11] >> 3;
    inputs[12] = preshift[12] >> 4;
    inputs[13] = preshift[13] >> 2;
    inputs[14] = preshift[14] >> 2;
    inputs[15] = preshift[15] >> 4;
    inputs[16] = preshift[16] >> 1;
    inputs[17] = preshift[17] >> 2;
    inputs[18] = preshift[18] >> 7;
    inputs[19] = preshift[19] >> 4;
    inputs[20] = preshift[20] >> 5;
    inputs[21] = preshift[21] >> 6;
    inputs[22] = preshift[22] >> 2;
    inputs[23] = preshift[23] >> 4;
    inputs[24] = preshift[24] >> 4;
    inputs[25] = preshift[25] >> 2;
    inputs[26] = preshift[26] >> 3;
    inputs[27] = preshift[27] >> 3;
    inputs[28] = preshift[28] >> 1;
    inputs[29] = preshift[29] >> 3;
    inputs[30] = preshift[30] >> 5;
    inputs[31] = preshift[31] >> 2;
    inputs[32] = preshift[32] >> 4;
    inputs[33] = preshift[33] >> 3;
    inputs[34] = preshift[34] >> 1;
    inputs[35] = preshift[35] >> 3;
    inputs[36] = preshift[36] >> 1;
    inputs[37] = preshift[37] >> 0;
    inputs[38] = preshift[38] >> 1;
    inputs[39] = preshift[39] >> -1;
    inputs[40] = preshift[40] >> -2;
    inputs[41] = preshift[41] >> -1;
    inputs[42] = preshift[42] >> 6;
    inputs[43] = preshift[43] >> 5;

    /*
    inputs[0] = in_jet_pt_0;
    inputs[1] = in_jet_eta_0;
    inputs[2] = in_jet_phi_0;
    inputs[3] = in_jet_pt_1;
    inputs[4] = in_jet_eta_1;
    inputs[5] = in_jet_phi_1;
    inputs[6] = in_jet_pt_2;
    inputs[7] = in_jet_eta_2;
    inputs[8] = in_jet_phi_2;
    inputs[9] = in_jet_pt_3;
    inputs[10] = in_jet_eta_3;
    inputs[11] = in_jet_phi_3;
    inputs[12] = in_jet_pt_4;
    inputs[13] = in_jet_eta_4;
    inputs[14] = in_jet_phi_4;
    inputs[15] = in_jet_pt_5;
    inputs[16] = in_jet_eta_5;
    inputs[17] = in_jet_phi_5;
    inputs[18] = in_etau_pt_0;
    inputs[19] = in_etau_eta_0;
    inputs[20] = in_etau_phi_0;
    inputs[21] = in_etau_pt_1;
    inputs[22] = in_etau_eta_1;
    inputs[23] = in_etau_phi_1;
    inputs[24] = in_etau_pt_2;
    inputs[25] = in_etau_eta_2;
    inputs[26] = in_etau_phi_2;
    inputs[27] = in_etau_pt_3;
    inputs[28] = in_etau_eta_3;
    inputs[29] = in_etau_phi_3;
    inputs[30] = in_mu_pt_0;
    inputs[31] = in_mu_eta_0;
    inputs[32] = in_mu_phi_0;
    inputs[33] = in_mu_pt_1;
    inputs[34] = in_mu_eta_1;
    inputs[35] = in_mu_phi_1;
    inputs[36] = in_mu_pt_2;
    inputs[37] = in_mu_eta_2;
    inputs[38] = in_mu_phi_2;
    inputs[39] = in_mu_pt_3;
    inputs[40] = in_mu_eta_3;
    inputs[41] = in_mu_phi_3;
    inputs[42] = in_met_pt_0;
    inputs[43] = in_met_phi_0;
    */

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
    input_t inputs[N_INPUT_1_1];
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

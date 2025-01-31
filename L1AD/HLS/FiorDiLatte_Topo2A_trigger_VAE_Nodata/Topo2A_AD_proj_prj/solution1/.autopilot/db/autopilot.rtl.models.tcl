set SynModuleInfo {
  {SRCNAME {dense_latency<ap_fixed<15, 6, 5, 3, 0>, ap_fixed<38, 8, 5, 3, 0>, config2>} MODELNAME dense_latency_ap_fixed_15_6_5_3_0_ap_fixed_38_8_5_3_0_config2_s RTLNAME Topo2A_AD_proj_dense_latency_ap_fixed_15_6_5_3_0_ap_fixed_38_8_5_3_0_config2_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_15s_5s_19_1_1 RTLNAME Topo2A_AD_proj_mul_15s_5s_19_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15s_5ns_19_1_1 RTLNAME Topo2A_AD_proj_mul_15s_5ns_19_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {linear<ap_fixed<38, 8, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, linear_config3>} MODELNAME linear_ap_fixed_38_8_5_3_0_ap_fixed_16_6_5_3_0_linear_config3_s RTLNAME Topo2A_AD_proj_linear_ap_fixed_38_8_5_3_0_ap_fixed_16_6_5_3_0_linear_config3_s}
  {SRCNAME {normalize<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<32, 7, 5, 3, 0>, config4>} MODELNAME normalize_ap_fixed_16_6_5_3_0_ap_fixed_32_7_5_3_0_config4_s RTLNAME Topo2A_AD_proj_normalize_ap_fixed_16_6_5_3_0_ap_fixed_32_7_5_3_0_config4_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_16s_8ns_23_1_1 RTLNAME Topo2A_AD_proj_mul_16s_8ns_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_16s_9ns_24_1_1 RTLNAME Topo2A_AD_proj_mul_16s_9ns_24_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_16s_10ns_25_1_1 RTLNAME Topo2A_AD_proj_mul_16s_10ns_25_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_16s_7ns_22_1_1 RTLNAME Topo2A_AD_proj_mul_16s_7ns_22_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<32, 7, 5, 3, 0>, ap_ufixed<15, 0, 4, 0, 0>, relu_config5>} MODELNAME relu_ap_fixed_32_7_5_3_0_ap_ufixed_15_0_4_0_0_relu_config5_s RTLNAME Topo2A_AD_proj_relu_ap_fixed_32_7_5_3_0_ap_ufixed_15_0_4_0_0_relu_config5_s}
  {SRCNAME {dense_latency<ap_ufixed<15, 0, 4, 0, 0>, ap_fixed<22, 3, 5, 3, 0>, config6>} MODELNAME dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_22_3_5_3_0_config6_s RTLNAME Topo2A_AD_proj_dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_22_3_5_3_0_config6_s}
  {SRCNAME {linear<ap_fixed<22, 3, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, linear_config7>} MODELNAME linear_ap_fixed_22_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config7_s RTLNAME Topo2A_AD_proj_linear_ap_fixed_22_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config7_s}
  {SRCNAME {normalize<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<26, 4, 5, 3, 0>, config8>} MODELNAME normalize_ap_fixed_16_6_5_3_0_ap_fixed_26_4_5_3_0_config8_s RTLNAME Topo2A_AD_proj_normalize_ap_fixed_16_6_5_3_0_ap_fixed_26_4_5_3_0_config8_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_16s_13ns_26_1_1 RTLNAME Topo2A_AD_proj_mul_16s_13ns_26_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<26, 4, 5, 3, 0>, ap_ufixed<15, 0, 4, 0, 0>, relu_config9>} MODELNAME relu_ap_fixed_26_4_5_3_0_ap_ufixed_15_0_4_0_0_relu_config9_s RTLNAME Topo2A_AD_proj_relu_ap_fixed_26_4_5_3_0_ap_ufixed_15_0_4_0_0_relu_config9_s}
  {SRCNAME {dense_latency<ap_ufixed<15, 0, 4, 0, 0>, ap_fixed<23, 3, 5, 3, 0>, config10>} MODELNAME dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_23_3_5_3_0_config10_s RTLNAME Topo2A_AD_proj_dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_23_3_5_3_0_config10_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_15ns_5ns_19_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_5ns_19_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {linear<ap_fixed<23, 3, 5, 3, 0>, ap_fixed<16, 6, 5, 3, 0>, linear_config11>} MODELNAME linear_ap_fixed_23_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config11_s RTLNAME Topo2A_AD_proj_linear_ap_fixed_23_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config11_s}
  {SRCNAME Topo2A_AD_proj MODELNAME Topo2A_AD_proj RTLNAME Topo2A_AD_proj IS_TOP 1}
}

set SynModuleInfo {
  {SRCNAME {dense_latency<ap_fixed<19, 11, 5, 3, 0>, ap_fixed<19, 11, 5, 3, 0>, config2>} MODELNAME dense_latency_ap_fixed_19_11_5_3_0_ap_fixed_19_11_5_3_0_config2_s RTLNAME Topo2A_AD_proj_dense_latency_ap_fixed_19_11_5_3_0_ap_fixed_19_11_5_3_0_config2_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_19s_6s_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_6s_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_19s_5s_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_5s_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_19s_6ns_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_6ns_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_19s_5ns_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_5ns_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_19s_7s_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_7s_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_19s_8s_23_1_1 RTLNAME Topo2A_AD_proj_mul_19s_8s_23_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {relu<ap_fixed<19, 11, 5, 3, 0>, ap_ufixed<15, 0, 4, 0, 0>, relu_config4>} MODELNAME relu_ap_fixed_19_11_5_3_0_ap_ufixed_15_0_4_0_0_relu_config4_s RTLNAME Topo2A_AD_proj_relu_ap_fixed_19_11_5_3_0_ap_ufixed_15_0_4_0_0_relu_config4_s}
  {SRCNAME {dense_latency<ap_ufixed<15, 0, 4, 0, 0>, ap_fixed<14, 6, 5, 3, 0>, config5>} MODELNAME dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_14_6_5_3_0_config5_s RTLNAME Topo2A_AD_proj_dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_14_6_5_3_0_config5_s
    SUBMODULES {
      {MODELNAME Topo2A_AD_proj_mul_15ns_5ns_19_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_5ns_19_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15ns_7ns_21_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_7ns_21_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15ns_6s_21_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_6s_21_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15ns_7s_22_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_7s_22_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15ns_5s_20_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_5s_20_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME Topo2A_AD_proj_mul_15ns_6ns_20_1_1 RTLNAME Topo2A_AD_proj_mul_15ns_6ns_20_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME {linear<ap_fixed<14, 6, 5, 3, 0>, ap_fixed<19, 11, 5, 3, 0>, linear_config6>} MODELNAME linear_ap_fixed_14_6_5_3_0_ap_fixed_19_11_5_3_0_linear_config6_s RTLNAME Topo2A_AD_proj_linear_ap_fixed_14_6_5_3_0_ap_fixed_19_11_5_3_0_linear_config6_s}
  {SRCNAME {relu<ap_fixed<19, 11, 5, 3, 0>, ap_ufixed<15, 0, 4, 0, 0>, relu_config7>} MODELNAME relu_ap_fixed_19_11_5_3_0_ap_ufixed_15_0_4_0_0_relu_config7_s RTLNAME Topo2A_AD_proj_relu_ap_fixed_19_11_5_3_0_ap_ufixed_15_0_4_0_0_relu_config7_s}
  {SRCNAME {dense_latency<ap_ufixed<15, 0, 4, 0, 0>, ap_fixed<12, 4, 5, 3, 0>, config8>} MODELNAME dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_12_4_5_3_0_config8_s RTLNAME Topo2A_AD_proj_dense_latency_ap_ufixed_15_0_4_0_0_ap_fixed_12_4_5_3_0_config8_s}
  {SRCNAME {linear<ap_fixed<12, 4, 5, 3, 0>, ap_fixed<19, 11, 5, 3, 0>, linear_config9>} MODELNAME linear_ap_fixed_12_4_5_3_0_ap_fixed_19_11_5_3_0_linear_config9_s RTLNAME Topo2A_AD_proj_linear_ap_fixed_12_4_5_3_0_ap_fixed_19_11_5_3_0_linear_config9_s}
  {SRCNAME Topo2A_AD_proj MODELNAME Topo2A_AD_proj RTLNAME Topo2A_AD_proj IS_TOP 1}
}

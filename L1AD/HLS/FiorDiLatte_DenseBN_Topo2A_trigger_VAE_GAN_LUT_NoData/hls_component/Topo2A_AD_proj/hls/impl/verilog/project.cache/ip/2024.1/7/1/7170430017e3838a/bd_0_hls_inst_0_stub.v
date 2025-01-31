// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1 (lin64) Build 5076996 Wed May 22 18:36:09 MDT 2024
// Date        : Sat Jan 25 15:12:43 2025
// Host        : rdsrv413 running 64-bit Ubuntu 20.04.6 LTS
// Command     : write_verilog -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ bd_0_hls_inst_0_stub.v
// Design      : bd_0_hls_inst_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xcvu9p-flga2104-2-e
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "Topo2A_AD_proj,Vivado 2024.1" *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix(ap_done, ap_idle, ap_ready, ap_start, ap_rst, 
  in_jet_pt_0, in_jet_eta_0, in_jet_phi_0, in_jet_pt_1, in_jet_eta_1, in_jet_phi_1, 
  in_jet_pt_2, in_jet_eta_2, in_jet_phi_2, in_jet_pt_3, in_jet_eta_3, in_jet_phi_3, 
  in_jet_pt_4, in_jet_eta_4, in_jet_phi_4, in_jet_pt_5, in_jet_eta_5, in_jet_phi_5, 
  in_etau_pt_0, in_etau_eta_0, in_etau_phi_0, in_etau_pt_1, in_etau_eta_1, in_etau_phi_1, 
  in_etau_pt_2, in_etau_eta_2, in_etau_phi_2, in_etau_pt_3, in_etau_eta_3, in_etau_phi_3, 
  in_mu_pt_0, in_mu_eta_0, in_mu_phi_0, in_mu_pt_1, in_mu_eta_1, in_mu_phi_1, in_mu_pt_2, 
  in_mu_eta_2, in_mu_phi_2, in_mu_pt_3, in_mu_eta_3, in_mu_phi_3, in_met_pt_0, in_met_phi_0, 
  layer9_out_0, layer9_out_1, layer9_out_2)
/* synthesis syn_black_box black_box_pad_pin="ap_done,ap_idle,ap_ready,ap_start,ap_rst,in_jet_pt_0[31:0],in_jet_eta_0[31:0],in_jet_phi_0[31:0],in_jet_pt_1[31:0],in_jet_eta_1[31:0],in_jet_phi_1[31:0],in_jet_pt_2[31:0],in_jet_eta_2[31:0],in_jet_phi_2[31:0],in_jet_pt_3[31:0],in_jet_eta_3[31:0],in_jet_phi_3[31:0],in_jet_pt_4[31:0],in_jet_eta_4[31:0],in_jet_phi_4[31:0],in_jet_pt_5[31:0],in_jet_eta_5[31:0],in_jet_phi_5[31:0],in_etau_pt_0[31:0],in_etau_eta_0[31:0],in_etau_phi_0[31:0],in_etau_pt_1[31:0],in_etau_eta_1[31:0],in_etau_phi_1[31:0],in_etau_pt_2[31:0],in_etau_eta_2[31:0],in_etau_phi_2[31:0],in_etau_pt_3[31:0],in_etau_eta_3[31:0],in_etau_phi_3[31:0],in_mu_pt_0[31:0],in_mu_eta_0[31:0],in_mu_phi_0[31:0],in_mu_pt_1[31:0],in_mu_eta_1[31:0],in_mu_phi_1[31:0],in_mu_pt_2[31:0],in_mu_eta_2[31:0],in_mu_phi_2[31:0],in_mu_pt_3[31:0],in_mu_eta_3[31:0],in_mu_phi_3[31:0],in_met_pt_0[31:0],in_met_phi_0[31:0],layer9_out_0[18:0],layer9_out_1[18:0],layer9_out_2[18:0]" */;
  output ap_done;
  output ap_idle;
  output ap_ready;
  input ap_start;
  input ap_rst;
  input [31:0]in_jet_pt_0;
  input [31:0]in_jet_eta_0;
  input [31:0]in_jet_phi_0;
  input [31:0]in_jet_pt_1;
  input [31:0]in_jet_eta_1;
  input [31:0]in_jet_phi_1;
  input [31:0]in_jet_pt_2;
  input [31:0]in_jet_eta_2;
  input [31:0]in_jet_phi_2;
  input [31:0]in_jet_pt_3;
  input [31:0]in_jet_eta_3;
  input [31:0]in_jet_phi_3;
  input [31:0]in_jet_pt_4;
  input [31:0]in_jet_eta_4;
  input [31:0]in_jet_phi_4;
  input [31:0]in_jet_pt_5;
  input [31:0]in_jet_eta_5;
  input [31:0]in_jet_phi_5;
  input [31:0]in_etau_pt_0;
  input [31:0]in_etau_eta_0;
  input [31:0]in_etau_phi_0;
  input [31:0]in_etau_pt_1;
  input [31:0]in_etau_eta_1;
  input [31:0]in_etau_phi_1;
  input [31:0]in_etau_pt_2;
  input [31:0]in_etau_eta_2;
  input [31:0]in_etau_phi_2;
  input [31:0]in_etau_pt_3;
  input [31:0]in_etau_eta_3;
  input [31:0]in_etau_phi_3;
  input [31:0]in_mu_pt_0;
  input [31:0]in_mu_eta_0;
  input [31:0]in_mu_phi_0;
  input [31:0]in_mu_pt_1;
  input [31:0]in_mu_eta_1;
  input [31:0]in_mu_phi_1;
  input [31:0]in_mu_pt_2;
  input [31:0]in_mu_eta_2;
  input [31:0]in_mu_phi_2;
  input [31:0]in_mu_pt_3;
  input [31:0]in_mu_eta_3;
  input [31:0]in_mu_phi_3;
  input [31:0]in_met_pt_0;
  input [31:0]in_met_phi_0;
  output [18:0]layer9_out_0;
  output [18:0]layer9_out_1;
  output [18:0]layer9_out_2;
endmodule

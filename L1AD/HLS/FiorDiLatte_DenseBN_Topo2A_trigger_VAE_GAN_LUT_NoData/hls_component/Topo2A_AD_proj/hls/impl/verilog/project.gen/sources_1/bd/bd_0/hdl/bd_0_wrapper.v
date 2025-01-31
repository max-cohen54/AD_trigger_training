//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2024.1 (lin64) Build 5076996 Wed May 22 18:36:09 MDT 2024
//Date        : Sat Jan 25 15:11:29 2025
//Host        : rdsrv413 running 64-bit Ubuntu 20.04.6 LTS
//Command     : generate_target bd_0_wrapper.bd
//Design      : bd_0_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module bd_0_wrapper
   (ap_ctrl_done,
    ap_ctrl_idle,
    ap_ctrl_ready,
    ap_ctrl_start,
    ap_rst,
    in_etau_eta_0,
    in_etau_eta_1,
    in_etau_eta_2,
    in_etau_eta_3,
    in_etau_phi_0,
    in_etau_phi_1,
    in_etau_phi_2,
    in_etau_phi_3,
    in_etau_pt_0,
    in_etau_pt_1,
    in_etau_pt_2,
    in_etau_pt_3,
    in_jet_eta_0,
    in_jet_eta_1,
    in_jet_eta_2,
    in_jet_eta_3,
    in_jet_eta_4,
    in_jet_eta_5,
    in_jet_phi_0,
    in_jet_phi_1,
    in_jet_phi_2,
    in_jet_phi_3,
    in_jet_phi_4,
    in_jet_phi_5,
    in_jet_pt_0,
    in_jet_pt_1,
    in_jet_pt_2,
    in_jet_pt_3,
    in_jet_pt_4,
    in_jet_pt_5,
    in_met_phi_0,
    in_met_pt_0,
    in_mu_eta_0,
    in_mu_eta_1,
    in_mu_eta_2,
    in_mu_eta_3,
    in_mu_phi_0,
    in_mu_phi_1,
    in_mu_phi_2,
    in_mu_phi_3,
    in_mu_pt_0,
    in_mu_pt_1,
    in_mu_pt_2,
    in_mu_pt_3,
    layer9_out_0,
    layer9_out_1,
    layer9_out_2);
  output ap_ctrl_done;
  output ap_ctrl_idle;
  output ap_ctrl_ready;
  input ap_ctrl_start;
  input ap_rst;
  input [31:0]in_etau_eta_0;
  input [31:0]in_etau_eta_1;
  input [31:0]in_etau_eta_2;
  input [31:0]in_etau_eta_3;
  input [31:0]in_etau_phi_0;
  input [31:0]in_etau_phi_1;
  input [31:0]in_etau_phi_2;
  input [31:0]in_etau_phi_3;
  input [31:0]in_etau_pt_0;
  input [31:0]in_etau_pt_1;
  input [31:0]in_etau_pt_2;
  input [31:0]in_etau_pt_3;
  input [31:0]in_jet_eta_0;
  input [31:0]in_jet_eta_1;
  input [31:0]in_jet_eta_2;
  input [31:0]in_jet_eta_3;
  input [31:0]in_jet_eta_4;
  input [31:0]in_jet_eta_5;
  input [31:0]in_jet_phi_0;
  input [31:0]in_jet_phi_1;
  input [31:0]in_jet_phi_2;
  input [31:0]in_jet_phi_3;
  input [31:0]in_jet_phi_4;
  input [31:0]in_jet_phi_5;
  input [31:0]in_jet_pt_0;
  input [31:0]in_jet_pt_1;
  input [31:0]in_jet_pt_2;
  input [31:0]in_jet_pt_3;
  input [31:0]in_jet_pt_4;
  input [31:0]in_jet_pt_5;
  input [31:0]in_met_phi_0;
  input [31:0]in_met_pt_0;
  input [31:0]in_mu_eta_0;
  input [31:0]in_mu_eta_1;
  input [31:0]in_mu_eta_2;
  input [31:0]in_mu_eta_3;
  input [31:0]in_mu_phi_0;
  input [31:0]in_mu_phi_1;
  input [31:0]in_mu_phi_2;
  input [31:0]in_mu_phi_3;
  input [31:0]in_mu_pt_0;
  input [31:0]in_mu_pt_1;
  input [31:0]in_mu_pt_2;
  input [31:0]in_mu_pt_3;
  output [18:0]layer9_out_0;
  output [18:0]layer9_out_1;
  output [18:0]layer9_out_2;

  wire ap_ctrl_done;
  wire ap_ctrl_idle;
  wire ap_ctrl_ready;
  wire ap_ctrl_start;
  wire ap_rst;
  wire [31:0]in_etau_eta_0;
  wire [31:0]in_etau_eta_1;
  wire [31:0]in_etau_eta_2;
  wire [31:0]in_etau_eta_3;
  wire [31:0]in_etau_phi_0;
  wire [31:0]in_etau_phi_1;
  wire [31:0]in_etau_phi_2;
  wire [31:0]in_etau_phi_3;
  wire [31:0]in_etau_pt_0;
  wire [31:0]in_etau_pt_1;
  wire [31:0]in_etau_pt_2;
  wire [31:0]in_etau_pt_3;
  wire [31:0]in_jet_eta_0;
  wire [31:0]in_jet_eta_1;
  wire [31:0]in_jet_eta_2;
  wire [31:0]in_jet_eta_3;
  wire [31:0]in_jet_eta_4;
  wire [31:0]in_jet_eta_5;
  wire [31:0]in_jet_phi_0;
  wire [31:0]in_jet_phi_1;
  wire [31:0]in_jet_phi_2;
  wire [31:0]in_jet_phi_3;
  wire [31:0]in_jet_phi_4;
  wire [31:0]in_jet_phi_5;
  wire [31:0]in_jet_pt_0;
  wire [31:0]in_jet_pt_1;
  wire [31:0]in_jet_pt_2;
  wire [31:0]in_jet_pt_3;
  wire [31:0]in_jet_pt_4;
  wire [31:0]in_jet_pt_5;
  wire [31:0]in_met_phi_0;
  wire [31:0]in_met_pt_0;
  wire [31:0]in_mu_eta_0;
  wire [31:0]in_mu_eta_1;
  wire [31:0]in_mu_eta_2;
  wire [31:0]in_mu_eta_3;
  wire [31:0]in_mu_phi_0;
  wire [31:0]in_mu_phi_1;
  wire [31:0]in_mu_phi_2;
  wire [31:0]in_mu_phi_3;
  wire [31:0]in_mu_pt_0;
  wire [31:0]in_mu_pt_1;
  wire [31:0]in_mu_pt_2;
  wire [31:0]in_mu_pt_3;
  wire [18:0]layer9_out_0;
  wire [18:0]layer9_out_1;
  wire [18:0]layer9_out_2;

  bd_0 bd_0_i
       (.ap_ctrl_done(ap_ctrl_done),
        .ap_ctrl_idle(ap_ctrl_idle),
        .ap_ctrl_ready(ap_ctrl_ready),
        .ap_ctrl_start(ap_ctrl_start),
        .ap_rst(ap_rst),
        .in_etau_eta_0(in_etau_eta_0),
        .in_etau_eta_1(in_etau_eta_1),
        .in_etau_eta_2(in_etau_eta_2),
        .in_etau_eta_3(in_etau_eta_3),
        .in_etau_phi_0(in_etau_phi_0),
        .in_etau_phi_1(in_etau_phi_1),
        .in_etau_phi_2(in_etau_phi_2),
        .in_etau_phi_3(in_etau_phi_3),
        .in_etau_pt_0(in_etau_pt_0),
        .in_etau_pt_1(in_etau_pt_1),
        .in_etau_pt_2(in_etau_pt_2),
        .in_etau_pt_3(in_etau_pt_3),
        .in_jet_eta_0(in_jet_eta_0),
        .in_jet_eta_1(in_jet_eta_1),
        .in_jet_eta_2(in_jet_eta_2),
        .in_jet_eta_3(in_jet_eta_3),
        .in_jet_eta_4(in_jet_eta_4),
        .in_jet_eta_5(in_jet_eta_5),
        .in_jet_phi_0(in_jet_phi_0),
        .in_jet_phi_1(in_jet_phi_1),
        .in_jet_phi_2(in_jet_phi_2),
        .in_jet_phi_3(in_jet_phi_3),
        .in_jet_phi_4(in_jet_phi_4),
        .in_jet_phi_5(in_jet_phi_5),
        .in_jet_pt_0(in_jet_pt_0),
        .in_jet_pt_1(in_jet_pt_1),
        .in_jet_pt_2(in_jet_pt_2),
        .in_jet_pt_3(in_jet_pt_3),
        .in_jet_pt_4(in_jet_pt_4),
        .in_jet_pt_5(in_jet_pt_5),
        .in_met_phi_0(in_met_phi_0),
        .in_met_pt_0(in_met_pt_0),
        .in_mu_eta_0(in_mu_eta_0),
        .in_mu_eta_1(in_mu_eta_1),
        .in_mu_eta_2(in_mu_eta_2),
        .in_mu_eta_3(in_mu_eta_3),
        .in_mu_phi_0(in_mu_phi_0),
        .in_mu_phi_1(in_mu_phi_1),
        .in_mu_phi_2(in_mu_phi_2),
        .in_mu_phi_3(in_mu_phi_3),
        .in_mu_pt_0(in_mu_pt_0),
        .in_mu_pt_1(in_mu_pt_1),
        .in_mu_pt_2(in_mu_pt_2),
        .in_mu_pt_3(in_mu_pt_3),
        .layer9_out_0(layer9_out_0),
        .layer9_out_1(layer9_out_1),
        .layer9_out_2(layer9_out_2));
endmodule

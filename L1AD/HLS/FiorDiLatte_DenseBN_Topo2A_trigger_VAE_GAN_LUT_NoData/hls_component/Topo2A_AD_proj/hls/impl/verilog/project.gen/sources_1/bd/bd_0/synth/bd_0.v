//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2024.1 (lin64) Build 5076996 Wed May 22 18:36:09 MDT 2024
//Date        : Sat Jan 25 15:11:29 2025
//Host        : rdsrv413 running 64-bit Ubuntu 20.04.6 LTS
//Command     : generate_target bd_0.bd
//Design      : bd_0
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CORE_GENERATION_INFO = "bd_0,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=bd_0,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=1,numReposBlks=1,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=1,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,synth_mode=Hierarchical}" *) (* HW_HANDOFF = "bd_0.hwdef" *) 
module bd_0
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
  (* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl " *) output ap_ctrl_done;
  (* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl " *) output ap_ctrl_idle;
  (* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl " *) output ap_ctrl_ready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl " *) input ap_ctrl_start;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST.AP_RST RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST.AP_RST, INSERT_VIP 0, POLARITY ACTIVE_HIGH" *) input ap_rst;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_ETA_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_ETA_0, LAYERED_METADATA undef" *) input [31:0]in_etau_eta_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_ETA_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_ETA_1, LAYERED_METADATA undef" *) input [31:0]in_etau_eta_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_ETA_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_ETA_2, LAYERED_METADATA undef" *) input [31:0]in_etau_eta_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_ETA_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_ETA_3, LAYERED_METADATA undef" *) input [31:0]in_etau_eta_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PHI_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PHI_0, LAYERED_METADATA undef" *) input [31:0]in_etau_phi_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PHI_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PHI_1, LAYERED_METADATA undef" *) input [31:0]in_etau_phi_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PHI_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PHI_2, LAYERED_METADATA undef" *) input [31:0]in_etau_phi_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PHI_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PHI_3, LAYERED_METADATA undef" *) input [31:0]in_etau_phi_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PT_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PT_0, LAYERED_METADATA undef" *) input [31:0]in_etau_pt_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PT_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PT_1, LAYERED_METADATA undef" *) input [31:0]in_etau_pt_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PT_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PT_2, LAYERED_METADATA undef" *) input [31:0]in_etau_pt_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_ETAU_PT_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_ETAU_PT_3, LAYERED_METADATA undef" *) input [31:0]in_etau_pt_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_0, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_1, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_2, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_3, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_4 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_4, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_4;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_ETA_5 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_ETA_5, LAYERED_METADATA undef" *) input [31:0]in_jet_eta_5;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_0, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_1, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_2, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_3, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_4 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_4, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_4;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PHI_5 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PHI_5, LAYERED_METADATA undef" *) input [31:0]in_jet_phi_5;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_0, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_1, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_2, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_3, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_4 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_4, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_4;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_JET_PT_5 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_JET_PT_5, LAYERED_METADATA undef" *) input [31:0]in_jet_pt_5;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MET_PHI_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MET_PHI_0, LAYERED_METADATA undef" *) input [31:0]in_met_phi_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MET_PT_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MET_PT_0, LAYERED_METADATA undef" *) input [31:0]in_met_pt_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_ETA_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_ETA_0, LAYERED_METADATA undef" *) input [31:0]in_mu_eta_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_ETA_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_ETA_1, LAYERED_METADATA undef" *) input [31:0]in_mu_eta_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_ETA_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_ETA_2, LAYERED_METADATA undef" *) input [31:0]in_mu_eta_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_ETA_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_ETA_3, LAYERED_METADATA undef" *) input [31:0]in_mu_eta_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PHI_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PHI_0, LAYERED_METADATA undef" *) input [31:0]in_mu_phi_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PHI_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PHI_1, LAYERED_METADATA undef" *) input [31:0]in_mu_phi_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PHI_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PHI_2, LAYERED_METADATA undef" *) input [31:0]in_mu_phi_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PHI_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PHI_3, LAYERED_METADATA undef" *) input [31:0]in_mu_phi_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PT_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PT_0, LAYERED_METADATA undef" *) input [31:0]in_mu_pt_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PT_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PT_1, LAYERED_METADATA undef" *) input [31:0]in_mu_pt_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PT_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PT_2, LAYERED_METADATA undef" *) input [31:0]in_mu_pt_2;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.IN_MU_PT_3 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.IN_MU_PT_3, LAYERED_METADATA undef" *) input [31:0]in_mu_pt_3;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.LAYER9_OUT_0 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.LAYER9_OUT_0, LAYERED_METADATA undef" *) output [18:0]layer9_out_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.LAYER9_OUT_1 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.LAYER9_OUT_1, LAYERED_METADATA undef" *) output [18:0]layer9_out_1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 DATA.LAYER9_OUT_2 DATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME DATA.LAYER9_OUT_2, LAYERED_METADATA undef" *) output [18:0]layer9_out_2;

  wire ap_ctrl_0_1_done;
  wire ap_ctrl_0_1_idle;
  wire ap_ctrl_0_1_ready;
  wire ap_ctrl_0_1_start;
  wire ap_rst_0_1;
  wire [18:0]hls_inst_layer9_out_0;
  wire [18:0]hls_inst_layer9_out_1;
  wire [18:0]hls_inst_layer9_out_2;
  wire [31:0]in_etau_eta_0_0_1;
  wire [31:0]in_etau_eta_1_0_1;
  wire [31:0]in_etau_eta_2_0_1;
  wire [31:0]in_etau_eta_3_0_1;
  wire [31:0]in_etau_phi_0_0_1;
  wire [31:0]in_etau_phi_1_0_1;
  wire [31:0]in_etau_phi_2_0_1;
  wire [31:0]in_etau_phi_3_0_1;
  wire [31:0]in_etau_pt_0_0_1;
  wire [31:0]in_etau_pt_1_0_1;
  wire [31:0]in_etau_pt_2_0_1;
  wire [31:0]in_etau_pt_3_0_1;
  wire [31:0]in_jet_eta_0_0_1;
  wire [31:0]in_jet_eta_1_0_1;
  wire [31:0]in_jet_eta_2_0_1;
  wire [31:0]in_jet_eta_3_0_1;
  wire [31:0]in_jet_eta_4_0_1;
  wire [31:0]in_jet_eta_5_0_1;
  wire [31:0]in_jet_phi_0_0_1;
  wire [31:0]in_jet_phi_1_0_1;
  wire [31:0]in_jet_phi_2_0_1;
  wire [31:0]in_jet_phi_3_0_1;
  wire [31:0]in_jet_phi_4_0_1;
  wire [31:0]in_jet_phi_5_0_1;
  wire [31:0]in_jet_pt_0_0_1;
  wire [31:0]in_jet_pt_1_0_1;
  wire [31:0]in_jet_pt_2_0_1;
  wire [31:0]in_jet_pt_3_0_1;
  wire [31:0]in_jet_pt_4_0_1;
  wire [31:0]in_jet_pt_5_0_1;
  wire [31:0]in_met_phi_0_0_1;
  wire [31:0]in_met_pt_0_0_1;
  wire [31:0]in_mu_eta_0_0_1;
  wire [31:0]in_mu_eta_1_0_1;
  wire [31:0]in_mu_eta_2_0_1;
  wire [31:0]in_mu_eta_3_0_1;
  wire [31:0]in_mu_phi_0_0_1;
  wire [31:0]in_mu_phi_1_0_1;
  wire [31:0]in_mu_phi_2_0_1;
  wire [31:0]in_mu_phi_3_0_1;
  wire [31:0]in_mu_pt_0_0_1;
  wire [31:0]in_mu_pt_1_0_1;
  wire [31:0]in_mu_pt_2_0_1;
  wire [31:0]in_mu_pt_3_0_1;

  assign ap_ctrl_0_1_start = ap_ctrl_start;
  assign ap_ctrl_done = ap_ctrl_0_1_done;
  assign ap_ctrl_idle = ap_ctrl_0_1_idle;
  assign ap_ctrl_ready = ap_ctrl_0_1_ready;
  assign ap_rst_0_1 = ap_rst;
  assign in_etau_eta_0_0_1 = in_etau_eta_0[31:0];
  assign in_etau_eta_1_0_1 = in_etau_eta_1[31:0];
  assign in_etau_eta_2_0_1 = in_etau_eta_2[31:0];
  assign in_etau_eta_3_0_1 = in_etau_eta_3[31:0];
  assign in_etau_phi_0_0_1 = in_etau_phi_0[31:0];
  assign in_etau_phi_1_0_1 = in_etau_phi_1[31:0];
  assign in_etau_phi_2_0_1 = in_etau_phi_2[31:0];
  assign in_etau_phi_3_0_1 = in_etau_phi_3[31:0];
  assign in_etau_pt_0_0_1 = in_etau_pt_0[31:0];
  assign in_etau_pt_1_0_1 = in_etau_pt_1[31:0];
  assign in_etau_pt_2_0_1 = in_etau_pt_2[31:0];
  assign in_etau_pt_3_0_1 = in_etau_pt_3[31:0];
  assign in_jet_eta_0_0_1 = in_jet_eta_0[31:0];
  assign in_jet_eta_1_0_1 = in_jet_eta_1[31:0];
  assign in_jet_eta_2_0_1 = in_jet_eta_2[31:0];
  assign in_jet_eta_3_0_1 = in_jet_eta_3[31:0];
  assign in_jet_eta_4_0_1 = in_jet_eta_4[31:0];
  assign in_jet_eta_5_0_1 = in_jet_eta_5[31:0];
  assign in_jet_phi_0_0_1 = in_jet_phi_0[31:0];
  assign in_jet_phi_1_0_1 = in_jet_phi_1[31:0];
  assign in_jet_phi_2_0_1 = in_jet_phi_2[31:0];
  assign in_jet_phi_3_0_1 = in_jet_phi_3[31:0];
  assign in_jet_phi_4_0_1 = in_jet_phi_4[31:0];
  assign in_jet_phi_5_0_1 = in_jet_phi_5[31:0];
  assign in_jet_pt_0_0_1 = in_jet_pt_0[31:0];
  assign in_jet_pt_1_0_1 = in_jet_pt_1[31:0];
  assign in_jet_pt_2_0_1 = in_jet_pt_2[31:0];
  assign in_jet_pt_3_0_1 = in_jet_pt_3[31:0];
  assign in_jet_pt_4_0_1 = in_jet_pt_4[31:0];
  assign in_jet_pt_5_0_1 = in_jet_pt_5[31:0];
  assign in_met_phi_0_0_1 = in_met_phi_0[31:0];
  assign in_met_pt_0_0_1 = in_met_pt_0[31:0];
  assign in_mu_eta_0_0_1 = in_mu_eta_0[31:0];
  assign in_mu_eta_1_0_1 = in_mu_eta_1[31:0];
  assign in_mu_eta_2_0_1 = in_mu_eta_2[31:0];
  assign in_mu_eta_3_0_1 = in_mu_eta_3[31:0];
  assign in_mu_phi_0_0_1 = in_mu_phi_0[31:0];
  assign in_mu_phi_1_0_1 = in_mu_phi_1[31:0];
  assign in_mu_phi_2_0_1 = in_mu_phi_2[31:0];
  assign in_mu_phi_3_0_1 = in_mu_phi_3[31:0];
  assign in_mu_pt_0_0_1 = in_mu_pt_0[31:0];
  assign in_mu_pt_1_0_1 = in_mu_pt_1[31:0];
  assign in_mu_pt_2_0_1 = in_mu_pt_2[31:0];
  assign in_mu_pt_3_0_1 = in_mu_pt_3[31:0];
  assign layer9_out_0[18:0] = hls_inst_layer9_out_0;
  assign layer9_out_1[18:0] = hls_inst_layer9_out_1;
  assign layer9_out_2[18:0] = hls_inst_layer9_out_2;
  bd_0_hls_inst_0 hls_inst
       (.ap_done(ap_ctrl_0_1_done),
        .ap_idle(ap_ctrl_0_1_idle),
        .ap_ready(ap_ctrl_0_1_ready),
        .ap_rst(ap_rst_0_1),
        .ap_start(ap_ctrl_0_1_start),
        .in_etau_eta_0(in_etau_eta_0_0_1),
        .in_etau_eta_1(in_etau_eta_1_0_1),
        .in_etau_eta_2(in_etau_eta_2_0_1),
        .in_etau_eta_3(in_etau_eta_3_0_1),
        .in_etau_phi_0(in_etau_phi_0_0_1),
        .in_etau_phi_1(in_etau_phi_1_0_1),
        .in_etau_phi_2(in_etau_phi_2_0_1),
        .in_etau_phi_3(in_etau_phi_3_0_1),
        .in_etau_pt_0(in_etau_pt_0_0_1),
        .in_etau_pt_1(in_etau_pt_1_0_1),
        .in_etau_pt_2(in_etau_pt_2_0_1),
        .in_etau_pt_3(in_etau_pt_3_0_1),
        .in_jet_eta_0(in_jet_eta_0_0_1),
        .in_jet_eta_1(in_jet_eta_1_0_1),
        .in_jet_eta_2(in_jet_eta_2_0_1),
        .in_jet_eta_3(in_jet_eta_3_0_1),
        .in_jet_eta_4(in_jet_eta_4_0_1),
        .in_jet_eta_5(in_jet_eta_5_0_1),
        .in_jet_phi_0(in_jet_phi_0_0_1),
        .in_jet_phi_1(in_jet_phi_1_0_1),
        .in_jet_phi_2(in_jet_phi_2_0_1),
        .in_jet_phi_3(in_jet_phi_3_0_1),
        .in_jet_phi_4(in_jet_phi_4_0_1),
        .in_jet_phi_5(in_jet_phi_5_0_1),
        .in_jet_pt_0(in_jet_pt_0_0_1),
        .in_jet_pt_1(in_jet_pt_1_0_1),
        .in_jet_pt_2(in_jet_pt_2_0_1),
        .in_jet_pt_3(in_jet_pt_3_0_1),
        .in_jet_pt_4(in_jet_pt_4_0_1),
        .in_jet_pt_5(in_jet_pt_5_0_1),
        .in_met_phi_0(in_met_phi_0_0_1),
        .in_met_pt_0(in_met_pt_0_0_1),
        .in_mu_eta_0(in_mu_eta_0_0_1),
        .in_mu_eta_1(in_mu_eta_1_0_1),
        .in_mu_eta_2(in_mu_eta_2_0_1),
        .in_mu_eta_3(in_mu_eta_3_0_1),
        .in_mu_phi_0(in_mu_phi_0_0_1),
        .in_mu_phi_1(in_mu_phi_1_0_1),
        .in_mu_phi_2(in_mu_phi_2_0_1),
        .in_mu_phi_3(in_mu_phi_3_0_1),
        .in_mu_pt_0(in_mu_pt_0_0_1),
        .in_mu_pt_1(in_mu_pt_1_0_1),
        .in_mu_pt_2(in_mu_pt_2_0_1),
        .in_mu_pt_3(in_mu_pt_3_0_1),
        .layer9_out_0(hls_inst_layer9_out_0),
        .layer9_out_1(hls_inst_layer9_out_1),
        .layer9_out_2(hls_inst_layer9_out_2));
endmodule

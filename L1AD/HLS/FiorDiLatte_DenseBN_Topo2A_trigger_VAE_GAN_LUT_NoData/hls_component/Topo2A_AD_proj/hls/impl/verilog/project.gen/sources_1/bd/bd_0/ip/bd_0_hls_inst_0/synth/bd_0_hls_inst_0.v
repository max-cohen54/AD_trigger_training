// (c) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// (c) Copyright 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
// 
// This file contains confidential and proprietary information
// of AMD and is protected under U.S. and international copyright
// and other intellectual property laws.
// 
// DISCLAIMER
// This disclaimer is not a license and does not grant any
// rights to the materials distributed herewith. Except as
// otherwise provided in a valid license issued to you by
// AMD, and to the maximum extent permitted by applicable
// law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// WITH ALL FAULTS, AND AMD HEREBY DISCLAIMS ALL WARRANTIES
// AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// (2) AMD shall not be liable (whether in contract or tort,
// including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature
// related to, arising under or in connection with these
// materials, including for any direct, or any indirect,
// special, incidental, or consequential loss or damage
// (including loss of data, profits, goodwill, or any type of
// loss or damage suffered as a result of any action brought
// by a third party) even if such damage or loss was
// reasonably foreseeable or AMD had been advised of the
// possibility of the same.
// 
// CRITICAL APPLICATIONS
// AMD products are not designed or intended to be fail-
// safe, or for use in any application requiring fail-safe
// performance, such as life-support or safety devices or
// systems, Class III medical devices, nuclear facilities,
// applications related to the deployment of airbags, or any
// other applications that could lead to death, personal
// injury, or severe property or environmental damage
// (individually and collectively, "Critical
// Applications"). Customer assumes the sole risk and
// liability of any use of AMD products in Critical
// Applications, subject only to applicable laws and
// regulations governing limitations on product liability.
// 
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// PART OF THIS FILE AT ALL TIMES.
// 
// DO NOT MODIFY THIS FILE.


// IP VLNV: xilinx.com:hls:Topo2A_AD_proj:1.0
// IP Revision: 2113926071

(* X_CORE_INFO = "Topo2A_AD_proj,Vivado 2024.1" *)
(* CHECK_LICENSE_TYPE = "bd_0_hls_inst_0,Topo2A_AD_proj,{}" *)
(* CORE_GENERATION_INFO = "bd_0_hls_inst_0,Topo2A_AD_proj,{x_ipProduct=Vivado 2024.1,x_ipVendor=xilinx.com,x_ipLibrary=hls,x_ipName=Topo2A_AD_proj,x_ipVersion=1.0,x_ipCoreRevision=2113926071,x_ipLanguage=VERILOG,x_ipSimLanguage=MIXED}" *)
(* IP_DEFINITION_SOURCE = "HLS" *)
(* DowngradeIPIdentifiedWarnings = "yes" *)
module bd_0_hls_inst_0 (
  ap_done,
  ap_idle,
  ap_ready,
  ap_start,
  ap_rst,
  in_jet_pt_0,
  in_jet_eta_0,
  in_jet_phi_0,
  in_jet_pt_1,
  in_jet_eta_1,
  in_jet_phi_1,
  in_jet_pt_2,
  in_jet_eta_2,
  in_jet_phi_2,
  in_jet_pt_3,
  in_jet_eta_3,
  in_jet_phi_3,
  in_jet_pt_4,
  in_jet_eta_4,
  in_jet_phi_4,
  in_jet_pt_5,
  in_jet_eta_5,
  in_jet_phi_5,
  in_etau_pt_0,
  in_etau_eta_0,
  in_etau_phi_0,
  in_etau_pt_1,
  in_etau_eta_1,
  in_etau_phi_1,
  in_etau_pt_2,
  in_etau_eta_2,
  in_etau_phi_2,
  in_etau_pt_3,
  in_etau_eta_3,
  in_etau_phi_3,
  in_mu_pt_0,
  in_mu_eta_0,
  in_mu_phi_0,
  in_mu_pt_1,
  in_mu_eta_1,
  in_mu_phi_1,
  in_mu_pt_2,
  in_mu_eta_2,
  in_mu_phi_2,
  in_mu_pt_3,
  in_mu_eta_3,
  in_mu_phi_3,
  in_met_pt_0,
  in_met_phi_0,
  layer9_out_0,
  layer9_out_1,
  layer9_out_2
);

(* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl done" *)
output wire ap_done;
(* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl idle" *)
output wire ap_idle;
(* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl ready" *)
output wire ap_ready;
(* X_INTERFACE_INFO = "xilinx.com:interface:acc_handshake:1.0 ap_ctrl start" *)
input wire ap_start;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst, POLARITY ACTIVE_HIGH, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst RST" *)
input wire ap_rst;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_0 DATA" *)
input wire [31 : 0] in_jet_pt_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_0 DATA" *)
input wire [31 : 0] in_jet_eta_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_0 DATA" *)
input wire [31 : 0] in_jet_phi_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_1 DATA" *)
input wire [31 : 0] in_jet_pt_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_1 DATA" *)
input wire [31 : 0] in_jet_eta_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_1 DATA" *)
input wire [31 : 0] in_jet_phi_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_2 DATA" *)
input wire [31 : 0] in_jet_pt_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_2 DATA" *)
input wire [31 : 0] in_jet_eta_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_2 DATA" *)
input wire [31 : 0] in_jet_phi_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_3 DATA" *)
input wire [31 : 0] in_jet_pt_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_3 DATA" *)
input wire [31 : 0] in_jet_eta_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_3 DATA" *)
input wire [31 : 0] in_jet_phi_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_4, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_4 DATA" *)
input wire [31 : 0] in_jet_pt_4;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_4, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_4 DATA" *)
input wire [31 : 0] in_jet_eta_4;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_4, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_4 DATA" *)
input wire [31 : 0] in_jet_phi_4;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_pt_5, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_pt_5 DATA" *)
input wire [31 : 0] in_jet_pt_5;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_eta_5, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_eta_5 DATA" *)
input wire [31 : 0] in_jet_eta_5;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_jet_phi_5, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_jet_phi_5 DATA" *)
input wire [31 : 0] in_jet_phi_5;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_pt_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_pt_0 DATA" *)
input wire [31 : 0] in_etau_pt_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_eta_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_eta_0 DATA" *)
input wire [31 : 0] in_etau_eta_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_phi_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_phi_0 DATA" *)
input wire [31 : 0] in_etau_phi_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_pt_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_pt_1 DATA" *)
input wire [31 : 0] in_etau_pt_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_eta_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_eta_1 DATA" *)
input wire [31 : 0] in_etau_eta_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_phi_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_phi_1 DATA" *)
input wire [31 : 0] in_etau_phi_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_pt_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_pt_2 DATA" *)
input wire [31 : 0] in_etau_pt_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_eta_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_eta_2 DATA" *)
input wire [31 : 0] in_etau_eta_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_phi_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_phi_2 DATA" *)
input wire [31 : 0] in_etau_phi_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_pt_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_pt_3 DATA" *)
input wire [31 : 0] in_etau_pt_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_eta_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_eta_3 DATA" *)
input wire [31 : 0] in_etau_eta_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_etau_phi_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_etau_phi_3 DATA" *)
input wire [31 : 0] in_etau_phi_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_pt_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_pt_0 DATA" *)
input wire [31 : 0] in_mu_pt_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_eta_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_eta_0 DATA" *)
input wire [31 : 0] in_mu_eta_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_phi_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_phi_0 DATA" *)
input wire [31 : 0] in_mu_phi_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_pt_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_pt_1 DATA" *)
input wire [31 : 0] in_mu_pt_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_eta_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_eta_1 DATA" *)
input wire [31 : 0] in_mu_eta_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_phi_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_phi_1 DATA" *)
input wire [31 : 0] in_mu_phi_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_pt_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_pt_2 DATA" *)
input wire [31 : 0] in_mu_pt_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_eta_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_eta_2 DATA" *)
input wire [31 : 0] in_mu_eta_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_phi_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_phi_2 DATA" *)
input wire [31 : 0] in_mu_phi_2;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_pt_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_pt_3 DATA" *)
input wire [31 : 0] in_mu_pt_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_eta_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_eta_3 DATA" *)
input wire [31 : 0] in_mu_eta_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_mu_phi_3, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_mu_phi_3 DATA" *)
input wire [31 : 0] in_mu_phi_3;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_met_pt_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_met_pt_0 DATA" *)
input wire [31 : 0] in_met_pt_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in_met_phi_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 in_met_phi_0 DATA" *)
input wire [31 : 0] in_met_phi_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME layer9_out_0, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 layer9_out_0 DATA" *)
output wire [18 : 0] layer9_out_0;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME layer9_out_1, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 layer9_out_1 DATA" *)
output wire [18 : 0] layer9_out_1;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME layer9_out_2, LAYERED_METADATA undef" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 layer9_out_2 DATA" *)
output wire [18 : 0] layer9_out_2;

(* SDX_KERNEL = "true" *)
(* SDX_KERNEL_TYPE = "hls" *)
(* SDX_KERNEL_SYNTH_INST = "inst" *)
  Topo2A_AD_proj inst (
    .ap_done(ap_done),
    .ap_idle(ap_idle),
    .ap_ready(ap_ready),
    .ap_start(ap_start),
    .ap_rst(ap_rst),
    .in_jet_pt_0(in_jet_pt_0),
    .in_jet_eta_0(in_jet_eta_0),
    .in_jet_phi_0(in_jet_phi_0),
    .in_jet_pt_1(in_jet_pt_1),
    .in_jet_eta_1(in_jet_eta_1),
    .in_jet_phi_1(in_jet_phi_1),
    .in_jet_pt_2(in_jet_pt_2),
    .in_jet_eta_2(in_jet_eta_2),
    .in_jet_phi_2(in_jet_phi_2),
    .in_jet_pt_3(in_jet_pt_3),
    .in_jet_eta_3(in_jet_eta_3),
    .in_jet_phi_3(in_jet_phi_3),
    .in_jet_pt_4(in_jet_pt_4),
    .in_jet_eta_4(in_jet_eta_4),
    .in_jet_phi_4(in_jet_phi_4),
    .in_jet_pt_5(in_jet_pt_5),
    .in_jet_eta_5(in_jet_eta_5),
    .in_jet_phi_5(in_jet_phi_5),
    .in_etau_pt_0(in_etau_pt_0),
    .in_etau_eta_0(in_etau_eta_0),
    .in_etau_phi_0(in_etau_phi_0),
    .in_etau_pt_1(in_etau_pt_1),
    .in_etau_eta_1(in_etau_eta_1),
    .in_etau_phi_1(in_etau_phi_1),
    .in_etau_pt_2(in_etau_pt_2),
    .in_etau_eta_2(in_etau_eta_2),
    .in_etau_phi_2(in_etau_phi_2),
    .in_etau_pt_3(in_etau_pt_3),
    .in_etau_eta_3(in_etau_eta_3),
    .in_etau_phi_3(in_etau_phi_3),
    .in_mu_pt_0(in_mu_pt_0),
    .in_mu_eta_0(in_mu_eta_0),
    .in_mu_phi_0(in_mu_phi_0),
    .in_mu_pt_1(in_mu_pt_1),
    .in_mu_eta_1(in_mu_eta_1),
    .in_mu_phi_1(in_mu_phi_1),
    .in_mu_pt_2(in_mu_pt_2),
    .in_mu_eta_2(in_mu_eta_2),
    .in_mu_phi_2(in_mu_phi_2),
    .in_mu_pt_3(in_mu_pt_3),
    .in_mu_eta_3(in_mu_eta_3),
    .in_mu_phi_3(in_mu_phi_3),
    .in_met_pt_0(in_met_pt_0),
    .in_met_phi_0(in_met_phi_0),
    .layer9_out_0(layer9_out_0),
    .layer9_out_1(layer9_out_1),
    .layer9_out_2(layer9_out_2)
  );
endmodule

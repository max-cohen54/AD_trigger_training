-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2024.1 (lin64) Build 5076996 Wed May 22 18:36:09 MDT 2024
-- Date        : Sat Jan 25 15:12:43 2025
-- Host        : rdsrv413 running 64-bit Ubuntu 20.04.6 LTS
-- Command     : write_vhdl -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ bd_0_hls_inst_0_stub.vhdl
-- Design      : bd_0_hls_inst_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xcvu9p-flga2104-2-e
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  Port ( 
    ap_done : out STD_LOGIC;
    ap_idle : out STD_LOGIC;
    ap_ready : out STD_LOGIC;
    ap_start : in STD_LOGIC;
    ap_rst : in STD_LOGIC;
    in_jet_pt_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_pt_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_pt_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_pt_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_pt_4 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_4 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_4 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_pt_5 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_eta_5 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_jet_phi_5 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_pt_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_eta_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_phi_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_pt_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_eta_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_phi_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_pt_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_eta_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_phi_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_pt_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_eta_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_etau_phi_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_pt_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_eta_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_phi_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_pt_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_eta_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_phi_1 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_pt_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_eta_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_phi_2 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_pt_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_eta_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_mu_phi_3 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_met_pt_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    in_met_phi_0 : in STD_LOGIC_VECTOR ( 31 downto 0 );
    layer9_out_0 : out STD_LOGIC_VECTOR ( 18 downto 0 );
    layer9_out_1 : out STD_LOGIC_VECTOR ( 18 downto 0 );
    layer9_out_2 : out STD_LOGIC_VECTOR ( 18 downto 0 )
  );

end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture stub of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "ap_done,ap_idle,ap_ready,ap_start,ap_rst,in_jet_pt_0[31:0],in_jet_eta_0[31:0],in_jet_phi_0[31:0],in_jet_pt_1[31:0],in_jet_eta_1[31:0],in_jet_phi_1[31:0],in_jet_pt_2[31:0],in_jet_eta_2[31:0],in_jet_phi_2[31:0],in_jet_pt_3[31:0],in_jet_eta_3[31:0],in_jet_phi_3[31:0],in_jet_pt_4[31:0],in_jet_eta_4[31:0],in_jet_phi_4[31:0],in_jet_pt_5[31:0],in_jet_eta_5[31:0],in_jet_phi_5[31:0],in_etau_pt_0[31:0],in_etau_eta_0[31:0],in_etau_phi_0[31:0],in_etau_pt_1[31:0],in_etau_eta_1[31:0],in_etau_phi_1[31:0],in_etau_pt_2[31:0],in_etau_eta_2[31:0],in_etau_phi_2[31:0],in_etau_pt_3[31:0],in_etau_eta_3[31:0],in_etau_phi_3[31:0],in_mu_pt_0[31:0],in_mu_eta_0[31:0],in_mu_phi_0[31:0],in_mu_pt_1[31:0],in_mu_eta_1[31:0],in_mu_phi_1[31:0],in_mu_pt_2[31:0],in_mu_eta_2[31:0],in_mu_phi_2[31:0],in_mu_pt_3[31:0],in_mu_eta_3[31:0],in_mu_phi_3[31:0],in_met_pt_0[31:0],in_met_phi_0[31:0],layer9_out_0[18:0],layer9_out_1[18:0],layer9_out_2[18:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "Topo2A_AD_proj,Vivado 2024.1";
begin
end;

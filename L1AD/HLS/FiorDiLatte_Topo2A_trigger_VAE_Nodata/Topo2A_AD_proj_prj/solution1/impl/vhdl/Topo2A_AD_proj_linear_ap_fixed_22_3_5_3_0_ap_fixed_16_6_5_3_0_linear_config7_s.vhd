-- ==============================================================
-- Generated by Vitis HLS v2024.1
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity Topo2A_AD_proj_linear_ap_fixed_22_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config7_s is
port (
    ap_ready : OUT STD_LOGIC;
    data_0_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_1_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_2_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_3_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_4_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_5_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_6_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_7_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_8_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_9_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_10_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_11_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_12_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_13_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_14_val : IN STD_LOGIC_VECTOR (21 downto 0);
    data_15_val : IN STD_LOGIC_VECTOR (21 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_4 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_5 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_6 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_7 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_8 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_9 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_10 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_11 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_12 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_13 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_14 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_15 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_rst : IN STD_LOGIC );
end;


architecture behav of Topo2A_AD_proj_linear_ap_fixed_22_3_5_3_0_ap_fixed_16_6_5_3_0_linear_config7_s is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv32_9 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001001";
    constant ap_const_lv32_15 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000010101";
    constant ap_const_logic_0 : STD_LOGIC := '0';

attribute shreg_extract : string;
    signal trunc_ln_fu_146_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_s_fu_160_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_32_fu_174_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_33_fu_188_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_34_fu_202_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_35_fu_216_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_36_fu_230_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_37_fu_244_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_38_fu_258_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_39_fu_272_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_40_fu_286_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_41_fu_300_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_42_fu_314_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_43_fu_328_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_44_fu_342_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln32_45_fu_356_p4 : STD_LOGIC_VECTOR (12 downto 0);
    signal sext_ln32_fu_156_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_3_fu_170_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_4_fu_184_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_5_fu_198_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_6_fu_212_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_7_fu_226_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_8_fu_240_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_9_fu_254_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_10_fu_268_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_11_fu_282_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_12_fu_296_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_13_fu_310_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_14_fu_324_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_15_fu_338_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_16_fu_352_p1 : STD_LOGIC_VECTOR (15 downto 0);
    signal sext_ln32_17_fu_366_p1 : STD_LOGIC_VECTOR (15 downto 0);


begin



    ap_ready <= ap_const_logic_1;
    ap_return_0 <= sext_ln32_fu_156_p1;
    ap_return_1 <= sext_ln32_3_fu_170_p1;
    ap_return_10 <= sext_ln32_12_fu_296_p1;
    ap_return_11 <= sext_ln32_13_fu_310_p1;
    ap_return_12 <= sext_ln32_14_fu_324_p1;
    ap_return_13 <= sext_ln32_15_fu_338_p1;
    ap_return_14 <= sext_ln32_16_fu_352_p1;
    ap_return_15 <= sext_ln32_17_fu_366_p1;
    ap_return_2 <= sext_ln32_4_fu_184_p1;
    ap_return_3 <= sext_ln32_5_fu_198_p1;
    ap_return_4 <= sext_ln32_6_fu_212_p1;
    ap_return_5 <= sext_ln32_7_fu_226_p1;
    ap_return_6 <= sext_ln32_8_fu_240_p1;
    ap_return_7 <= sext_ln32_9_fu_254_p1;
    ap_return_8 <= sext_ln32_10_fu_268_p1;
    ap_return_9 <= sext_ln32_11_fu_282_p1;
        sext_ln32_10_fu_268_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_38_fu_258_p4),16));

        sext_ln32_11_fu_282_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_39_fu_272_p4),16));

        sext_ln32_12_fu_296_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_40_fu_286_p4),16));

        sext_ln32_13_fu_310_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_41_fu_300_p4),16));

        sext_ln32_14_fu_324_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_42_fu_314_p4),16));

        sext_ln32_15_fu_338_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_43_fu_328_p4),16));

        sext_ln32_16_fu_352_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_44_fu_342_p4),16));

        sext_ln32_17_fu_366_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_45_fu_356_p4),16));

        sext_ln32_3_fu_170_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_s_fu_160_p4),16));

        sext_ln32_4_fu_184_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_32_fu_174_p4),16));

        sext_ln32_5_fu_198_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_33_fu_188_p4),16));

        sext_ln32_6_fu_212_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_34_fu_202_p4),16));

        sext_ln32_7_fu_226_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_35_fu_216_p4),16));

        sext_ln32_8_fu_240_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_36_fu_230_p4),16));

        sext_ln32_9_fu_254_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln32_37_fu_244_p4),16));

        sext_ln32_fu_156_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(trunc_ln_fu_146_p4),16));

    trunc_ln32_32_fu_174_p4 <= data_2_val(21 downto 9);
    trunc_ln32_33_fu_188_p4 <= data_3_val(21 downto 9);
    trunc_ln32_34_fu_202_p4 <= data_4_val(21 downto 9);
    trunc_ln32_35_fu_216_p4 <= data_5_val(21 downto 9);
    trunc_ln32_36_fu_230_p4 <= data_6_val(21 downto 9);
    trunc_ln32_37_fu_244_p4 <= data_7_val(21 downto 9);
    trunc_ln32_38_fu_258_p4 <= data_8_val(21 downto 9);
    trunc_ln32_39_fu_272_p4 <= data_9_val(21 downto 9);
    trunc_ln32_40_fu_286_p4 <= data_10_val(21 downto 9);
    trunc_ln32_41_fu_300_p4 <= data_11_val(21 downto 9);
    trunc_ln32_42_fu_314_p4 <= data_12_val(21 downto 9);
    trunc_ln32_43_fu_328_p4 <= data_13_val(21 downto 9);
    trunc_ln32_44_fu_342_p4 <= data_14_val(21 downto 9);
    trunc_ln32_45_fu_356_p4 <= data_15_val(21 downto 9);
    trunc_ln32_s_fu_160_p4 <= data_1_val(21 downto 9);
    trunc_ln_fu_146_p4 <= data_0_val(21 downto 9);
end behav;
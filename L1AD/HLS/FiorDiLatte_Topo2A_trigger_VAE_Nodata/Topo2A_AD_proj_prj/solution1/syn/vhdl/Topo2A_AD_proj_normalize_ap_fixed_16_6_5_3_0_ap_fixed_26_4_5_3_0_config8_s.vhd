-- ==============================================================
-- Generated by Vitis HLS v2024.1
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity Topo2A_AD_proj_normalize_ap_fixed_16_6_5_3_0_ap_fixed_26_4_5_3_0_config8_s is
port (
    ap_ready : OUT STD_LOGIC;
    data_0_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_1_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_2_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_3_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_4_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_5_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_6_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_7_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_8_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_9_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_10_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_11_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_12_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_13_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_14_val : IN STD_LOGIC_VECTOR (15 downto 0);
    data_15_val : IN STD_LOGIC_VECTOR (15 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_4 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_5 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_6 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_7 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_8 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_9 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_10 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_11 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_12 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_13 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_14 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_return_15 : OUT STD_LOGIC_VECTOR (25 downto 0);
    ap_rst : IN STD_LOGIC );
end;


architecture behav of Topo2A_AD_proj_normalize_ap_fixed_16_6_5_3_0_ap_fixed_26_4_5_3_0_config8_s is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv26_EA0 : STD_LOGIC_VECTOR (25 downto 0) := "00000000000000111010100000";
    constant ap_const_lv26_B50 : STD_LOGIC_VECTOR (25 downto 0) := "00000000000000101101010000";
    constant ap_const_lv26_B48 : STD_LOGIC_VECTOR (25 downto 0) := "00000000000000101101001000";
    constant ap_const_lv26_F30 : STD_LOGIC_VECTOR (25 downto 0) := "00000000000000111100110000";
    constant ap_const_lv26_40000 : STD_LOGIC_VECTOR (25 downto 0) := "00000001000000000000000000";
    constant ap_const_lv26_18000 : STD_LOGIC_VECTOR (25 downto 0) := "00000000011000000000000000";
    constant ap_const_lv26_60000 : STD_LOGIC_VECTOR (25 downto 0) := "00000001100000000000000000";
    constant ap_const_lv26_1C000 : STD_LOGIC_VECTOR (25 downto 0) := "00000000011100000000000000";
    constant ap_const_lv26_54000 : STD_LOGIC_VECTOR (25 downto 0) := "00000001010100000000000000";
    constant ap_const_lv26_3FBC000 : STD_LOGIC_VECTOR (25 downto 0) := "11111110111100000000000000";
    constant ap_const_lv26_3FC8000 : STD_LOGIC_VECTOR (25 downto 0) := "11111111001000000000000000";
    constant ap_const_lv26_B0000 : STD_LOGIC_VECTOR (25 downto 0) := "00000010110000000000000000";
    constant ap_const_lv26_58000 : STD_LOGIC_VECTOR (25 downto 0) := "00000001011000000000000000";
    constant ap_const_lv26_B4000 : STD_LOGIC_VECTOR (25 downto 0) := "00000010110100000000000000";
    constant ap_const_lv26_64000 : STD_LOGIC_VECTOR (25 downto 0) := "00000001100100000000000000";
    constant ap_const_lv26_F4000 : STD_LOGIC_VECTOR (25 downto 0) := "00000011110100000000000000";
    constant ap_const_lv26_BC000 : STD_LOGIC_VECTOR (25 downto 0) := "00000010111100000000000000";
    constant ap_const_lv26_3F8C000 : STD_LOGIC_VECTOR (25 downto 0) := "11111110001100000000000000";
    constant ap_const_logic_0 : STD_LOGIC := '0';

attribute shreg_extract : string;
    signal mul_ln54_46_fu_182_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_fu_183_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_40_fu_184_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_35_fu_185_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_36_fu_186_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_32_fu_187_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_37_fu_188_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_42_fu_189_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_34_fu_190_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_39_fu_191_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_41_fu_192_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_43_fu_193_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_38_fu_194_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_44_fu_195_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_45_fu_196_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_33_fu_197_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal mul_ln54_fu_183_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_32_fu_187_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_33_fu_197_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_34_fu_190_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_35_fu_185_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_36_fu_186_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_37_fu_188_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_38_fu_194_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_39_fu_191_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_40_fu_184_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_41_fu_192_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_42_fu_189_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_43_fu_193_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_44_fu_195_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_45_fu_196_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal mul_ln54_46_fu_182_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_fu_1340_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_32_fu_1351_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_33_fu_1362_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_34_fu_1373_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_35_fu_1384_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_36_fu_1395_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_37_fu_1406_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_38_fu_1417_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_39_fu_1428_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_40_fu_1439_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_41_fu_1450_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_42_fu_1461_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_43_fu_1472_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_44_fu_1483_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_45_fu_1494_p2 : STD_LOGIC_VECTOR (25 downto 0);
    signal add_ln54_46_fu_1505_p2 : STD_LOGIC_VECTOR (25 downto 0);

    component Topo2A_AD_proj_mul_16s_13ns_26_1_1 IS
    generic (
        ID : INTEGER;
        NUM_STAGE : INTEGER;
        din0_WIDTH : INTEGER;
        din1_WIDTH : INTEGER;
        dout_WIDTH : INTEGER );
    port (
        din0 : IN STD_LOGIC_VECTOR (15 downto 0);
        din1 : IN STD_LOGIC_VECTOR (12 downto 0);
        dout : OUT STD_LOGIC_VECTOR (25 downto 0) );
    end component;



begin
    mul_16s_13ns_26_1_1_U188 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_15_val,
        din1 => mul_ln54_46_fu_182_p1,
        dout => mul_ln54_46_fu_182_p2);

    mul_16s_13ns_26_1_1_U189 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_0_val,
        din1 => mul_ln54_fu_183_p1,
        dout => mul_ln54_fu_183_p2);

    mul_16s_13ns_26_1_1_U190 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_9_val,
        din1 => mul_ln54_40_fu_184_p1,
        dout => mul_ln54_40_fu_184_p2);

    mul_16s_13ns_26_1_1_U191 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_4_val,
        din1 => mul_ln54_35_fu_185_p1,
        dout => mul_ln54_35_fu_185_p2);

    mul_16s_13ns_26_1_1_U192 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_5_val,
        din1 => mul_ln54_36_fu_186_p1,
        dout => mul_ln54_36_fu_186_p2);

    mul_16s_13ns_26_1_1_U193 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_1_val,
        din1 => mul_ln54_32_fu_187_p1,
        dout => mul_ln54_32_fu_187_p2);

    mul_16s_13ns_26_1_1_U194 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_6_val,
        din1 => mul_ln54_37_fu_188_p1,
        dout => mul_ln54_37_fu_188_p2);

    mul_16s_13ns_26_1_1_U195 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_11_val,
        din1 => mul_ln54_42_fu_189_p1,
        dout => mul_ln54_42_fu_189_p2);

    mul_16s_13ns_26_1_1_U196 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_3_val,
        din1 => mul_ln54_34_fu_190_p1,
        dout => mul_ln54_34_fu_190_p2);

    mul_16s_13ns_26_1_1_U197 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_8_val,
        din1 => mul_ln54_39_fu_191_p1,
        dout => mul_ln54_39_fu_191_p2);

    mul_16s_13ns_26_1_1_U198 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_10_val,
        din1 => mul_ln54_41_fu_192_p1,
        dout => mul_ln54_41_fu_192_p2);

    mul_16s_13ns_26_1_1_U199 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_12_val,
        din1 => mul_ln54_43_fu_193_p1,
        dout => mul_ln54_43_fu_193_p2);

    mul_16s_13ns_26_1_1_U200 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_7_val,
        din1 => mul_ln54_38_fu_194_p1,
        dout => mul_ln54_38_fu_194_p2);

    mul_16s_13ns_26_1_1_U201 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_13_val,
        din1 => mul_ln54_44_fu_195_p1,
        dout => mul_ln54_44_fu_195_p2);

    mul_16s_13ns_26_1_1_U202 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_14_val,
        din1 => mul_ln54_45_fu_196_p1,
        dout => mul_ln54_45_fu_196_p2);

    mul_16s_13ns_26_1_1_U203 : component Topo2A_AD_proj_mul_16s_13ns_26_1_1
    generic map (
        ID => 1,
        NUM_STAGE => 1,
        din0_WIDTH => 16,
        din1_WIDTH => 13,
        dout_WIDTH => 26)
    port map (
        din0 => data_2_val,
        din1 => mul_ln54_33_fu_197_p1,
        dout => mul_ln54_33_fu_197_p2);




    add_ln54_32_fu_1351_p2 <= std_logic_vector(unsigned(mul_ln54_32_fu_187_p2) + unsigned(ap_const_lv26_18000));
    add_ln54_33_fu_1362_p2 <= std_logic_vector(unsigned(mul_ln54_33_fu_197_p2) + unsigned(ap_const_lv26_60000));
    add_ln54_34_fu_1373_p2 <= std_logic_vector(unsigned(mul_ln54_34_fu_190_p2) + unsigned(ap_const_lv26_1C000));
    add_ln54_35_fu_1384_p2 <= std_logic_vector(unsigned(mul_ln54_35_fu_185_p2) + unsigned(ap_const_lv26_54000));
    add_ln54_36_fu_1395_p2 <= std_logic_vector(unsigned(mul_ln54_36_fu_186_p2) + unsigned(ap_const_lv26_3FBC000));
    add_ln54_37_fu_1406_p2 <= std_logic_vector(unsigned(mul_ln54_37_fu_188_p2) + unsigned(ap_const_lv26_3FC8000));
    add_ln54_38_fu_1417_p2 <= std_logic_vector(unsigned(mul_ln54_38_fu_194_p2) + unsigned(ap_const_lv26_B0000));
    add_ln54_39_fu_1428_p2 <= std_logic_vector(unsigned(mul_ln54_39_fu_191_p2) + unsigned(ap_const_lv26_58000));
    add_ln54_40_fu_1439_p2 <= std_logic_vector(unsigned(mul_ln54_40_fu_184_p2) + unsigned(ap_const_lv26_58000));
    add_ln54_41_fu_1450_p2 <= std_logic_vector(unsigned(mul_ln54_41_fu_192_p2) + unsigned(ap_const_lv26_B4000));
    add_ln54_42_fu_1461_p2 <= std_logic_vector(unsigned(mul_ln54_42_fu_189_p2) + unsigned(ap_const_lv26_B0000));
    add_ln54_43_fu_1472_p2 <= std_logic_vector(unsigned(mul_ln54_43_fu_193_p2) + unsigned(ap_const_lv26_64000));
    add_ln54_44_fu_1483_p2 <= std_logic_vector(unsigned(mul_ln54_44_fu_195_p2) + unsigned(ap_const_lv26_F4000));
    add_ln54_45_fu_1494_p2 <= std_logic_vector(unsigned(mul_ln54_45_fu_196_p2) + unsigned(ap_const_lv26_BC000));
    add_ln54_46_fu_1505_p2 <= std_logic_vector(unsigned(mul_ln54_46_fu_182_p2) + unsigned(ap_const_lv26_3F8C000));
    add_ln54_fu_1340_p2 <= std_logic_vector(unsigned(mul_ln54_fu_183_p2) + unsigned(ap_const_lv26_40000));
    ap_ready <= ap_const_logic_1;
    ap_return_0 <= add_ln54_fu_1340_p2;
    ap_return_1 <= add_ln54_32_fu_1351_p2;
    ap_return_10 <= add_ln54_41_fu_1450_p2;
    ap_return_11 <= add_ln54_42_fu_1461_p2;
    ap_return_12 <= add_ln54_43_fu_1472_p2;
    ap_return_13 <= add_ln54_44_fu_1483_p2;
    ap_return_14 <= add_ln54_45_fu_1494_p2;
    ap_return_15 <= add_ln54_46_fu_1505_p2;
    ap_return_2 <= add_ln54_33_fu_1362_p2;
    ap_return_3 <= add_ln54_34_fu_1373_p2;
    ap_return_4 <= add_ln54_35_fu_1384_p2;
    ap_return_5 <= add_ln54_36_fu_1395_p2;
    ap_return_6 <= add_ln54_37_fu_1406_p2;
    ap_return_7 <= add_ln54_38_fu_1417_p2;
    ap_return_8 <= add_ln54_39_fu_1428_p2;
    ap_return_9 <= add_ln54_40_fu_1439_p2;
    mul_ln54_32_fu_187_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
    mul_ln54_33_fu_197_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_34_fu_190_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
    mul_ln54_35_fu_185_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_36_fu_186_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
    mul_ln54_37_fu_188_p1 <= ap_const_lv26_F30(13 - 1 downto 0);
    mul_ln54_38_fu_194_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_39_fu_191_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_40_fu_184_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_41_fu_192_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_42_fu_189_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
    mul_ln54_43_fu_193_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_44_fu_195_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
    mul_ln54_45_fu_196_p1 <= ap_const_lv26_B48(13 - 1 downto 0);
    mul_ln54_46_fu_182_p1 <= ap_const_lv26_EA0(13 - 1 downto 0);
    mul_ln54_fu_183_p1 <= ap_const_lv26_B50(13 - 1 downto 0);
end behav;
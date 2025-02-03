# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
# Tool Version Limit: 2024.05
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
# 
# ==============================================================
#
# Settings for Vivado implementation flow
#
set top_module Topo2A_AD_proj
set language verilog
set family virtexuplus
set device xcvu9p
set package -flga2104
set speed -2-e
set clock ""
set fsm_ext "off"

# For customizing the implementation flow
set add_io_buffers false ;# true|false

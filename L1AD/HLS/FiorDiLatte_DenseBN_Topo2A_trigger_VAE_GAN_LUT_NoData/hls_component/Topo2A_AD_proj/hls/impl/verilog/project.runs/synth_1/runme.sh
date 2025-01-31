#!/bin/sh

# 
# Vivado(TM)
# runme.sh: a Vivado-generated Runs Script for UNIX
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
# 

if [ -z "$PATH" ]; then
  PATH=/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vivado/2024.1/bin
else
  PATH=/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vitis/2024.1/bin:/afs/slac.stanford.edu/g/reseng/vol39/xilinx/2024.1/Vivado/2024.1/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/u1/hjia625/conifer/FiorDiLatte_DenseBN_Topo2A_trigger_VAE_GAN_LUT_NoData/hls_component/Topo2A_AD_proj/hls/impl/verilog/project.runs/synth_1'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

EAStep vivado -log bd_0_wrapper.vds -m64 -product Vivado -mode batch -messageDb vivado.pb -notrace -source bd_0_wrapper.tcl

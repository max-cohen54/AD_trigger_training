#ifndef TOPO2A_AD_PROJ_H_
#define TOPO2A_AD_PROJ_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void Topo2A_AD_proj(
    input_t inputs[N_INPUT_1_1],
    result_t layer11_out[N_LAYER_10]
);

#endif

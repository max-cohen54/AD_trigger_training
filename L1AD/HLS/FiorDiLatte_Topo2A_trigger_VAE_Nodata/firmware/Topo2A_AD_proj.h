#ifndef TOPO2A_AD_PROJ_H_
#define TOPO2A_AD_PROJ_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void Topo2A_AD_proj(
    int in_jet_pt_0,
    int in_jet_eta_0,
    int in_jet_phi_0,
    int in_jet_pt_1,
    int in_jet_eta_1,
    int in_jet_phi_1,
    int in_jet_pt_2,
    int in_jet_eta_2,
    int in_jet_phi_2,
    int in_jet_pt_3,
    int in_jet_eta_3,
    int in_jet_phi_3,
    int in_jet_pt_4,
    int in_jet_eta_4,
    int in_jet_phi_4,
    int in_jet_pt_5,
    int in_jet_eta_5,
    int in_jet_phi_5,
    int in_etau_pt_0,
    int in_etau_eta_0,
    int in_etau_phi_0,
    int in_etau_pt_1,
    int in_etau_eta_1,
    int in_etau_phi_1,
    int in_etau_pt_2,
    int in_etau_eta_2,
    int in_etau_phi_2,
    int in_etau_pt_3,
    int in_etau_eta_3,
    int in_etau_phi_3,
    int in_mu_pt_0,
    int in_mu_eta_0,
    int in_mu_phi_0,
    int in_mu_pt_1,
    int in_mu_eta_1,
    int in_mu_phi_1,
    int in_mu_pt_2,
    int in_mu_eta_2,
    int in_mu_phi_2,
    int in_mu_pt_3,
    int in_mu_eta_3,
    int in_mu_phi_3,
    int in_met_pt_0,
    int in_met_phi_0,
    result_t layer11_out[N_LAYER_10]
);

#endif

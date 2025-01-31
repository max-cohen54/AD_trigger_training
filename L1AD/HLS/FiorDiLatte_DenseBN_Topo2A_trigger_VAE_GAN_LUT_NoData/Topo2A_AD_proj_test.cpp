#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/Topo2A_AD_proj.h"
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {
    // load input data from text file
    std::ifstream fin("tb_data/raw_data/Topo_2A_ChiPlusChiMinus500_40_10ns_pure.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/raw_predictions/Topo_2A_ChiPlusChiMinus500_40_10ns_prediction.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/Topo_2A_ChiPlusChiMinus500_40_10ns_rtlsim.log";
#else
    std::string RESULTS_LOG = "tb_data/Topo_2A_ChiPlusChiMinus500_40_10ns_csim.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<int> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

     	    // hls-fpga-machine-learning insert data
       	    int inputs[N_INPUT_1_1];
            for (int i=0; i < N_INPUT_1_1; i++) {
                inputs[i] = in[i];
            }
	    int in_jet_pt_0 = inputs[0];
            int in_jet_eta_0 = inputs[1];
            int in_jet_phi_0 = inputs[2];
            int in_jet_pt_1 = inputs[3];
            int in_jet_eta_1 = inputs[4];
            int in_jet_phi_1 = inputs[5];
            int in_jet_pt_2 = inputs[6];
            int in_jet_eta_2 = inputs[7];
            int in_jet_phi_2 = inputs[8];
            int in_jet_pt_3 = inputs[9];
            int in_jet_eta_3 = inputs[10];
            int in_jet_phi_3 = inputs[11];
            int in_jet_pt_4 = inputs[12];
            int in_jet_eta_4 = inputs[13];
            int in_jet_phi_4 = inputs[14];
            int in_jet_pt_5 = inputs[15];
            int in_jet_eta_5 = inputs[16];
            int in_jet_phi_5 = inputs[17];
            int in_etau_pt_0 = inputs[18];
            int in_etau_eta_0 = inputs[19];
            int in_etau_phi_0 = inputs[20];
            int in_etau_pt_1 = inputs[21];
            int in_etau_eta_1 = inputs[22];
            int in_etau_phi_1 = inputs[23];
            int in_etau_pt_2 = inputs[24];
            int in_etau_eta_2 = inputs[25];
            int in_etau_phi_2 = inputs[26];
            int in_etau_pt_3 = inputs[27];
            int in_etau_eta_3 = inputs[28];
            int in_etau_phi_3 = inputs[29];
            int in_mu_pt_0 = inputs[30];
            int in_mu_eta_0 = inputs[31];
            int in_mu_phi_0 = inputs[32];
            int in_mu_pt_1 = inputs[33];
            int in_mu_eta_1 = inputs[34];
            int in_mu_phi_1 = inputs[35];
            int in_mu_pt_2 = inputs[36];
            int in_mu_eta_2 = inputs[37];
            int in_mu_phi_2 = inputs[38];
            int in_mu_pt_3 = inputs[39];
            int in_mu_eta_3 = inputs[40];
            int in_mu_phi_3 = inputs[41];
            int in_met_pt_0 = inputs[42];
            int in_met_phi_0 = inputs[43];
            result_t layer9_out[N_LAYER_8];

            // hls-fpga-machine-learning insert top-level-function
            Topo2A_AD_proj(in_jet_pt_0,in_jet_eta_0,in_jet_phi_0,in_jet_pt_1,in_jet_eta_1,in_jet_phi_1,in_jet_pt_2,in_jet_eta_2,in_jet_phi_2,in_jet_pt_3,in_jet_eta_3,in_jet_phi_3,in_jet_pt_4,in_jet_eta_4,in_jet_phi_4,in_jet_pt_5,in_jet_eta_5,in_jet_phi_5,in_etau_pt_0,in_etau_eta_0,in_etau_phi_0,in_etau_pt_1,in_etau_eta_1,in_etau_phi_1,in_etau_pt_2,in_etau_eta_2,in_etau_phi_2,in_etau_pt_3,in_etau_eta_3,in_etau_phi_3,in_mu_pt_0,in_mu_eta_0,in_mu_phi_0,in_mu_pt_1,in_mu_eta_1,in_mu_phi_1,in_mu_pt_2,in_mu_eta_2,in_mu_phi_2,in_mu_pt_3,in_mu_eta_3,in_mu_phi_3,in_met_pt_0,in_met_phi_0,layer9_out);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for(int i = 0; i < N_LAYER_8; i++) {
                  std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, N_LAYER_8>(layer9_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, N_LAYER_8>(layer9_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero
        int inputs[N_INPUT_1_1];
        nnet::fill_zero<int, N_INPUT_1_1>(inputs);
        int in_jet_pt_0 = inputs[0];
        int in_jet_eta_0 = inputs[1];
        int in_jet_phi_0 = inputs[2];
        int in_jet_pt_1 = inputs[3];
        int in_jet_eta_1 = inputs[4];
        int in_jet_phi_1 = inputs[5];
        int in_jet_pt_2 = inputs[6];
        int in_jet_eta_2 = inputs[7];
        int in_jet_phi_2 = inputs[8];
        int in_jet_pt_3 = inputs[9];
        int in_jet_eta_3 = inputs[10];
        int in_jet_phi_3 = inputs[11];
        int in_jet_pt_4 = inputs[12];
        int in_jet_eta_4 = inputs[13];
        int in_jet_phi_4 = inputs[14];
        int in_jet_pt_5 = inputs[15];
        int in_jet_eta_5 = inputs[16];
        int in_jet_phi_5 = inputs[17];
        int in_etau_pt_0 = inputs[18];
        int in_etau_eta_0 = inputs[19];
        int in_etau_phi_0 = inputs[20];
        int in_etau_pt_1 = inputs[21];
        int in_etau_eta_1 = inputs[22];
        int in_etau_phi_1 = inputs[23];
        int in_etau_pt_2 = inputs[24];
        int in_etau_eta_2 = inputs[25];
        int in_etau_phi_2 = inputs[26];
        int in_etau_pt_3 = inputs[27];
        int in_etau_eta_3 = inputs[28];
        int in_etau_phi_3 = inputs[29];
        int in_mu_pt_0 = inputs[30];
        int in_mu_eta_0 = inputs[31];
        int in_mu_phi_0 = inputs[32];
        int in_mu_pt_1 = inputs[33];
        int in_mu_eta_1 = inputs[34];
        int in_mu_phi_1 = inputs[35];
        int in_mu_pt_2 = inputs[36];
        int in_mu_eta_2 = inputs[37];
        int in_mu_phi_2 = inputs[38];
        int in_mu_pt_3 = inputs[39];
        int in_mu_eta_3 = inputs[40];
        int in_mu_phi_3 = inputs[41];
        int in_met_pt_0 = inputs[42];
        int in_met_phi_0 = inputs[43];
        result_t layer9_out[N_LAYER_8];
        
        // hls-fpga-machine-learning insert top-level-function
        Topo2A_AD_proj(in_jet_pt_0,in_jet_eta_0,in_jet_phi_0,in_jet_pt_1,in_jet_eta_1,in_jet_phi_1,in_jet_pt_2,in_jet_eta_2,in_jet_phi_2,in_jet_pt_3,in_jet_eta_3,in_jet_phi_3,in_jet_pt_4,in_jet_eta_4,in_jet_phi_4,in_jet_pt_5,in_jet_eta_5,in_jet_phi_5,in_etau_pt_0,in_etau_eta_0,in_etau_phi_0,in_etau_pt_1,in_etau_eta_1,in_etau_phi_1,in_etau_pt_2,in_etau_eta_2,in_etau_phi_2,in_etau_pt_3,in_etau_eta_3,in_etau_phi_3,in_mu_pt_0,in_mu_eta_0,in_mu_phi_0,in_mu_pt_1,in_mu_eta_1,in_mu_phi_1,in_mu_pt_2,in_mu_eta_2,in_mu_phi_2,in_mu_pt_3,in_mu_eta_3,in_mu_phi_3,in_met_pt_0,in_met_phi_0,layer9_out);

        // hls-fpga-machine-learning insert output
        nnet::print_result<result_t, N_LAYER_8>(layer9_out, std::cout, true);

        // hls-fpga-machine-learning insert tb-output
        nnet::print_result<result_t, N_LAYER_8>(layer9_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}

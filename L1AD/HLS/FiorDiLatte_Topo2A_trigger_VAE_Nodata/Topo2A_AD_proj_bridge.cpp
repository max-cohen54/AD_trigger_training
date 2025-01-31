#ifndef TOPO2A_AD_PROJ_BRIDGE_H_
#define TOPO2A_AD_PROJ_BRIDGE_H_

#include "firmware/Topo2A_AD_proj.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense1", (void *) malloc(N_LAYER_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense1_linear", (void *) malloc(N_LAYER_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("BN1", (void *) malloc(N_LAYER_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("relu1", (void *) malloc(N_LAYER_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense2", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense2_linear", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("BN2", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("relu2", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("z_mean", (void *) malloc(N_LAYER_10 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("z_mean_linear", (void *) malloc(N_LAYER_10 * element_size)));
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void Topo2A_AD_proj_float(
    float inputs[N_INPUT_1_1],
    float layer11_out[N_LAYER_10]
) {

    input_t inputs_ap[N_INPUT_1_1];
    nnet::convert_data<float, input_t, N_INPUT_1_1>(inputs, inputs_ap);

    result_t layer11_out_ap[N_LAYER_10];

    Topo2A_AD_proj(inputs_ap,layer11_out_ap);

    nnet::convert_data<result_t, float, N_LAYER_10>(layer11_out_ap, layer11_out);
}

void Topo2A_AD_proj_double(
    double inputs[N_INPUT_1_1],
    double layer11_out[N_LAYER_10]
) {
    input_t inputs_ap[N_INPUT_1_1];
    nnet::convert_data<double, input_t, N_INPUT_1_1>(inputs, inputs_ap);

    result_t layer11_out_ap[N_LAYER_10];

    Topo2A_AD_proj(inputs_ap,layer11_out_ap);

    nnet::convert_data<result_t, double, N_LAYER_10>(layer11_out_ap, layer11_out);
}
}

#endif

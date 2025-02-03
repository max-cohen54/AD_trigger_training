import hls4ml
import matplotlib.pyplot as plt
import plotting
from scipy.special import softmax, expit as sigmoid
from sklearn import metrics as sk
from keras.models import load_model
import keras
import os
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
print(tf.__version__)
print(keras.__version__)
print(tfmot.__version__)

import tensorflow as tf
import tensorflow.math as tfmath
import tensorflow.keras as keras
from scipy.optimize import curve_fit
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as sk
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU, Input, LSTM, Flatten, Concatenate, Dense, Conv2D, TimeDistributed, MaxPooling2D, ReLU, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Precision
from qkeras import QActivation, QDense, QConv2D, QBatchNormalization, QConv2DBatchnorm
from qkeras import QDenseBatchnorm
from qkeras import quantized_relu, quantized_bits

#os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
os.environ['PATH'] = os.environ['XILINX_AP_INCLUDE'] + '/bin:' + os.environ['PATH']
with h5py.File('preprocessed_2A_data_FDL_GAN_ALT.h5', 'r') as hf:
    Topo_2A_L1failed = hf['Topo_2A_L1failed'][:]
    Topo_EB_L1failed_weights = hf['Topo_EB_L1failed_weights'][:]
    Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure = hf['Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure'][:]
    Topo_2A_ChiPlusChiMinus500_40_10ns_pure = hf['Topo_2A_ChiPlusChiMinus500_40_10ns_pure'][:]
    Topo_2A_HAHMggf_23e_pure = hf['Topo_2A_HAHMggf_23e_pure'][:]
    Topo_2A_HNLeemu_23e_pure = hf['Topo_2A_HNLeemu_23e_pure'][:]
    Topo_2A_jz2_23e_pure = hf['Topo_2A_jz2_23e_pure'][:]
    Topo_2A_jz4_23e_pure = hf['Topo_2A_jz4_23e_pure'][:]
    Topo_2A_HH4b_23e_pure = hf['Topo_2A_HH4b_23e_pure'][:]
    Topo_2A_ttbar_1lep_23e_pure = hf['Topo_2A_ttbar_1lep_23e_pure'][:]
    Topo_2A_ttbar_2lep_23e_pure = hf['Topo_2A_ttbar_2lep_23e_pure'][:]
    Topo_2A_ZZ4lep_23e_pure = hf['Topo_2A_ZZ4lep_23e_pure'][:]

    L1_pass_flag = hf['L1_pass_flag'][:]

print("Data loaded from preprocessed_2A_data_FDL_GAN.h5")
model_path = '2A_AE_model_FDL_GAN_ALT_23e.h5'
custom_objects = {
    'QDense': QDense,
    'QActivation': QActivation,
    'QDenseBatchnorm': QDenseBatchnorm,
    'quantized_relu': quantized_relu,
    'quantized_bits': quantized_bits,
}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
model.save("FiorDiLatte_FoldBN_GAN_ALT.h5")
print(model.summary())
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
for layer in model.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}")
        if len(layer.weights) > 0:
            print("  Weights:")
            print(layer.weights[0])
        if len(layer.weights) > 1:
            print("  Biases:")
            print(layer.weights[1])
        print("\n")

np.set_printoptions(threshold=100, linewidth=10)
############
# First, the baseline model
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend="Vitis", default_precision='ap_fixed<10,10>')
print("--------------------------------------------------------------------------------------")
print(hls_config)
print("--------------------------------------------------------------------------------------")
hls_config['Model']['ReuseFactor'] = 1 

for layer in hls_config['LayerName'].keys():
    print(layer)
    hls_config['LayerName'][layer]['Trace'] = True
    hls_config['LayerName'][layer]['Strategy'] = 'Latency'
    
    if layer == 'inputs':
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<10,10>'
    if layer == 'dense1':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['result'] =  'ap_fixed<10,10>'#'ap_fixed<19,11>''ap_fixed<27,11>'
    #if layer == 'dense1_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    #if layer == 'relu1':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == 'dense2':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<10,10>'#'ap_fixed<14,6>' 'ap_fixed<22,6>'
    #if layer == 'dense2_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    #if layer == 'relu2':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == 'z_mean':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<10,6>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<10,10>'#'ap_fixed<12,4>' 'ap_fixed<20,4>'
    #if layer == 'z_mean_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'

print("-------------------------------------------------------------------------")
print(hls_config)
print("-------------------------------------------------------------------------")


cfg = hls4ml.converters.create_config(backend='Vitis')

cfg['IOType'] = 'io_parallel'
cfg['HLSConfig'] = hls_config
cfg['KerasModel'] = model
cfg['OutputDir'] = 'FiorDiLatte_DenseBN_Topo2A_trigger_VAE_GAN_trash_NoData'
cfg['ClockPeriod'] = 25
cfg['ClockUncertainty'] = '27%'
cfg['Part'] = 'xcvu9p-flga2104-2-e'
cfg['ProjectName'] = 'Topo2A_AD_proj'

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()
hls_model.build(reset=False, csim=False, synth=False, cosim=False, validation=False, export=False, vsynth=False)
def AD_score_CKL(z_mean):
    return np.mean(z_mean**2, axis=-1)

def calculate_loss(model, data):
    z_mean = model.predict(np.ascontiguousarray(data))
    return AD_score_CKL(z_mean)
import struct
class Model_Evaluator:
    def __init__(
        self,
        model,
        backround,
        br_weights,
        signal,
        signal_weights,
        input_dim,
        title="placeholder",
        save=False,
        labels=None,
    ):
        self.input_dim = input_dim
        self.model = model  # Now directly using the model object
        self.signal = signal
        self.backround = backround
        self.br_loss = []
        self.signal_loss = []
        self.title = title
        self.saveplots = save
        self.labels = labels
        self.br_weights = br_weights
        self.signal_weights = signal_weights

    def calculate_loss(self, l_type):
        if l_type == "CKL":
            self.br_loss = calculate_loss(self.model, self.backround)
            self.signal_loss = [calculate_loss(self.model, batch) for batch in self.signal]
        else:
            raise ValueError(f"Unsupported loss type: {l_type}")

    def ROC(self):
        target_fpr = 3.125e-5/0.9956957918921533
        tpr_at_target = []
        thresholds_at_target = []
        plt.figure(figsize=(10, 8))
        plt.plot((32000000*0.9956957918921533)*np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal")
        
        br_loss = np.atleast_1d(self.br_loss)
        for j, batch_loss in enumerate(self.signal_loss):
            sig_w = self.signal_weights[j]
            br_w = self.br_weights
            weights = np.concatenate((br_w, sig_w))
            truth = np.concatenate([np.zeros(len(self.br_loss)), np.ones(len(batch_loss))])
            ROC_data = np.concatenate((self.br_loss, batch_loss))
            fpr, tpr, thresholds = sk.roc_curve(truth, ROC_data, sample_weight=weights)
            auc = sk.roc_auc_score(truth, ROC_data)
            plt.plot(fpr*32000000*0.9956957918921533, tpr, label=f"{self.labels[j]}: AUC = {auc:.3f}")

            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_target.append(tpr[idx])
            thresholds_at_target.append(thresholds[idx])

            with h5py.File(f"{self.labels[j]}_2A.h5", 'w') as hf:
                hf.create_dataset("threshold", data=thresholds[idx])
                hf.create_dataset("background_scores", data=self.br_loss)
                hf.create_dataset("signal_scores", data=batch_loss)
        plt.xlabel("Pure Rate")
        plt.ylabel("True Positive Rate")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{self.title} ROC")
        plt.vlines(1000, 0, 1, colors="r", linestyles="dashed")
        plt.legend(loc="lower right")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        if self.saveplots:
            plt.savefig(f"{self.title}_purerate_ROC.png", format="png", bbox_inches="tight", dpi=1200)
        plt.show()
        print(f"\nTPR at FPR = {target_fpr} for each channel:")
        for label, tpr, threshold in zip(self.labels, tpr_at_target, thresholds_at_target):
            print(f"{label}: {100*tpr:.6f}%, Theshold = {threshold:.30f}")
            print(''.join(f'{byte:08b}' for byte in struct.pack('!d', threshold)))

def plot_distribution_outside(br_loss, br_weights, signal_losses, signal_weights, labels, 
                                    title, saveplots=False, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    bins = 1000
    
    # Determine the range for the bins
    all_scores = np.concatenate([br_loss] + signal_losses)
    min_score, max_score = 0.01, np.max(all_scores)
    bin_edges = np.linspace(min_score, max_score, bins + 1)

    # Plot signal distributions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(signal_losses)))
    for i, (signal_scores, signal_weights, label) in enumerate(zip(signal_losses, signal_weights, labels)):
        signal_weights = signal_weights / np.sum(signal_weights)  # Normalize weights
        ax.hist(signal_scores, bins=bin_edges, weights=signal_weights, 
                histtype='step', label=label, color=colors[i],
                density=True, linewidth=2)
    
    # Plot background distribution
    br_weights = br_weights / np.sum(br_weights)  # Normalize weights
    ax.hist(br_loss, bins=bin_edges, weights=br_weights, histtype='step', 
            label='Background', color='black', density=True, linewidth=2)
    
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Weighted Normalized Density')
    ax.set_title(f'{title} Weighted Normalized Anomaly Score Distribution')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=1e-6, top=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if saveplots and save_path:
        plt.savefig(save_path, format="png", bbox_inches="tight")
    plt.show()        

def evaluate_model(model, title, is_hls_model=False):
    evaluator = Model_Evaluator(
        model=model,
        backround=Topo_2A_L1failed,
        br_weights=Topo_EB_L1failed_weights,
        signal=[
            Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure,
            Topo_2A_ChiPlusChiMinus500_40_10ns_pure,
            Topo_2A_HAHMggf_23e_pure,
            Topo_2A_HNLeemu_23e_pure,
            Topo_2A_jz2_23e_pure,
            Topo_2A_HH4b_23e_pure,
            Topo_2A_ttbar_1lep_23e_pure,
            Topo_2A_ttbar_2lep_23e_pure,
            Topo_2A_ZZ4lep_23e_pure
        ],
        signal_weights=[
            np.ones(len(Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure)),
            np.ones(len(Topo_2A_ChiPlusChiMinus500_40_10ns_pure)),
            np.ones(len(Topo_2A_HAHMggf_23e_pure)),
            np.ones(len(Topo_2A_HNLeemu_23e_pure)),
            np.ones(len(Topo_2A_jz2_23e_pure)),
            np.ones(len(Topo_2A_HH4b_23e_pure)),
            np.ones(len(Topo_2A_ttbar_1lep_23e_pure)),
            np.ones(len(Topo_2A_ttbar_2lep_23e_pure)),
            np.ones(len(Topo_2A_ZZ4lep_23e_pure))
        ],
        input_dim=Topo_2A_L1failed.shape[1],  # Adjust this if your input dimension is different
        title=title,
        save=True,  # Set to True if you want to save the ROC plots
        labels=[
            'ChiPlusChiMinus100_99_0p3ns_23e',
            'ChiPlusChiMinus500_40_10ns_23e',
            'HAHMggf_23e',
            'HNLeemu_23e',
            'jz2_23e',
            'HH4b_23e',
            'ttbar_1lep_23e',
            'ttbar_2lep_23e',
            'ZZ4lep_23e'
        ]
    )

    print(f"\nEvaluating {title}")
    if not is_hls_model:
        print("Model summary:")
        model.summary()

    else:
        print("HLS4ML model - summary not available")

    evaluator.calculate_loss("CKL")
    evaluator.ROC()
# Evaluate original Keras model
print("Evaluating original Keras model:")
#evaluate_model(model, "Original_Topo_2A_Model")

# Evaluate hls4ml model
print("Evaluating hls4ml model:")
evaluate_model(hls_model, "hls4ml_Topo_2A_Model", is_hls_model=True)
name_list = [
    "Topo_2A_L1failed",
    "Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure",
    "Topo_2A_ChiPlusChiMinus500_40_10ns_pure",
    "Topo_2A_HAHMggf_23e_pure",
    "Topo_2A_HNLeemu_23e_pure",
    "Topo_2A_jz2_23e_pure",
    "Topo_2A_HH4b_23e_pure",
    "Topo_2A_ttbar_1lep_23e_pure",
    "Topo_2A_ttbar_2lep_23e_pure",
    "Topo_2A_ZZ4lep_23e_pure",
]   
data_list = [
    Topo_2A_ChiPlusChiMinus100_99_0p3ns_pure,
    Topo_2A_ChiPlusChiMinus500_40_10ns_pure,
    Topo_2A_HAHMggf_23e_pure,
    Topo_2A_HNLeemu_23e_pure,
    Topo_2A_jz2_23e_pure,
    Topo_2A_HH4b_23e_pure,
    Topo_2A_ttbar_1lep_23e_pure,
    Topo_2A_ttbar_2lep_23e_pure,
    Topo_2A_ZZ4lep_23e_pure
]

for data in data_list:
    plots = hls4ml.model.profiling.numerical(model = model, hls_model = hls_model, X = data, plot='boxplot')
    plt.show()

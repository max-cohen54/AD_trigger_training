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
with h5py.File('preprocessed_2A_data_FDL_final.h5', 'r') as hf:
    Topo_2A_test_br = hf['Topo_2A_test_br'][:]
    Topo_br_weights = hf['Topo_br_weights'][:]
    Topo_2A_L1failed = hf['Topo_2A_L1failed'][:]
    Topo_EB_L1failed_weights = hf['Topo_EB_L1failed_weights'][:]
    Topo_2A_test_signal = hf['Topo_2A_test_signal'][:]
    Topo_2A_HLT_passed = hf['Topo_2A_HLT_passed'][:]
    Topo_2A_L1_passed = hf['Topo_2A_L1_passed'][:]
    Topo_2A_HHbbtt = hf['Topo_2A_HHbbtt'][:]
    Topo_2A_jz1 = hf['Topo_2A_jz1'][:]
    Topo_2A_jz2 = hf['Topo_2A_jz2'][:]
    Topo_2A_jz4 = hf['Topo_2A_jz4'][:]
    Topo_2A_A14 = hf['Topo_2A_A14'][:]
    Topo_2A_HAHMggf = hf['Topo_2A_HAHMggf'][:]
    Topo_2A_qqa = hf['Topo_2A_qqa'][:]
    Topo_2A_Zprime = hf['Topo_2A_Zprime'][:]
    Topo_2A_ZZ4lep = hf['Topo_2A_ZZ4lep'][:]
    Topo_2A_HHbbtt_pure = hf['Topo_2A_HHbbtt_pure'][:]
    Topo_2A_jz1_pure = hf['Topo_2A_jz1_pure'][:]
    Topo_2A_jz2_pure = hf['Topo_2A_jz2_pure'][:]
    Topo_2A_A14_pure = hf['Topo_2A_A14_pure'][:]
    Topo_2A_HAHMggf_pure = hf['Topo_2A_HAHMggf_pure'][:]
    Topo_2A_qqa_pure = hf['Topo_2A_qqa_pure'][:]
    Topo_2A_Zprime_pure = hf['Topo_2A_Zprime_pure'][:]
    Topo_2A_ZZ4lep_pure = hf['Topo_2A_ZZ4lep_pure'][:]

    Topo_signal_weights = hf['Topo_signal_weights'][:]
    HLT_pass_weights = hf['HLT_pass_weights'][:]
    L1_pass_weights = hf['L1_pass_weights'][:]
    HLT_pass_test = hf['HLT_pass_test'][:]
    L1_pass_test = hf['L1_pass_test'][:]

print("Data loaded from preprocessed_2A_data_FDL_final.h5")
model_path = '2A_AE_model_FDL_DenseBN_BESTOFLONGRUN'
custom_objects = {
    'QDense': QDense,
    'QActivation': QActivation,
    'QDenseBatchnorm': QDenseBatchnorm
}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#model.save("FiorDiLatte_FoldBN.keras")
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
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
print("--------------------------------------------------------------------------------------")
print(hls_config)
print("--------------------------------------------------------------------------------------")
hls_config['Model']['ReuseFactor'] = 1 
for layer in hls_config['LayerName'].keys():
    print(layer)
    hls_config['LayerName'][layer]['Trace'] = True
    hls_config['LayerName'][layer]['Strategy'] = 'Latency'
    
    if layer == 'inputs':
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<15,6>'
    if layer == 'dense1':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<38,8>'
    #if layer == 'dense1_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    #if layer == 'relu1':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == 'dense2':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<22,3>'
    #if layer == 'dense2_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    #if layer == 'relu2':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == 'z_mean':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<6,2>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<6,2>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<23,3>'
    #if layer == 'z_mean_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
print("-------------------------------------------------------------------------")
print(hls_config)
print("-------------------------------------------------------------------------")


cfg = hls4ml.converters.create_config(backend='Vitis')

cfg['IOType'] = 'io_parallel'
cfg['HLSConfig'] = hls_config
cfg['KerasModel'] = model
cfg['OutputDir'] = 'FiorDiLatte_DenseBN_Topo2A_trigger_VAE'
cfg['ClockPeriod'] = 25
cfg['ClockUncertainty'] = '27%'
cfg['Part'] = 'xcvu9p-flga2104-2-e'
cfg['ProjectName'] = 'Topo2A_AD_proj'

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()
hls_model.build(csim=False)
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
        target_fpr = 2.5e-5/(1-0.18456571428571428)
        tpr_at_target = []
        thresholds_at_target = []
        plt.figure(figsize=(10, 8))
        plt.plot((40000000*(1-0.18456571428571428))*np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal")
        
        br_loss = np.atleast_1d(self.br_loss)
        for j, batch_loss in enumerate(self.signal_loss):
            sig_w = self.signal_weights[j]
            br_w = self.br_weights
            weights = np.concatenate((br_w, sig_w))
            truth = np.concatenate([np.zeros(len(self.br_loss)), np.ones(len(batch_loss))])
            ROC_data = np.concatenate((self.br_loss, batch_loss))
            fpr, tpr, thresholds = sk.roc_curve(truth, ROC_data, sample_weight=weights)
            auc = sk.roc_auc_score(truth, ROC_data)
            plt.plot(fpr*40000000*(1-0.18456571428571428), tpr, label=f"{self.labels[j]}: AUC = {auc:.3f}")

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
            Topo_2A_HHbbtt_pure,
            Topo_2A_jz1_pure,
            Topo_2A_jz2_pure,
            Topo_2A_A14_pure,
            Topo_2A_HAHMggf_pure,
            Topo_2A_qqa_pure,
            Topo_2A_Zprime_pure,
            Topo_2A_ZZ4lep_pure
        ],
        signal_weights=[
            np.ones(len(Topo_2A_HHbbtt_pure)),
            np.ones(len(Topo_2A_jz1_pure)),
            np.ones(len(Topo_2A_jz2_pure)),
            np.ones(len(Topo_2A_A14_pure)),
            np.ones(len(Topo_2A_HAHMggf_pure)),
            np.ones(len(Topo_2A_qqa_pure)),
            np.ones(len(Topo_2A_Zprime_pure)),
            np.ones(len(Topo_2A_ZZ4lep_pure))
        ],
        input_dim=Topo_2A_L1failed.shape[1],  # Adjust this if your input dimension is different
        title=title,
        save=True,  # Set to True if you want to save the ROC plots
        labels=[
            'HH->bbtt', 'dijets#1', 'dijets#2',
            'HNL', 'HAHMggfZdZd', 'qqa', 'Zprime', 'ZZ4lep'
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
    "Topo_2A_HHbbtt_pure",
    "Topo_2A_A14_pure",
    "Topo_2A_HAHMggf_pure",
    "Topo_2A_qqa_pure",
    "Topo_2A_Zprime_pure",
    "Topo_2A_ZZ4lep_pure"
]   
data_list = [
    Topo_2A_L1failed,
    Topo_2A_HHbbtt_pure,
    Topo_2A_A14_pure,
    Topo_2A_HAHMggf_pure,
    Topo_2A_qqa_pure,
    Topo_2A_Zprime_pure,
    Topo_2A_ZZ4lep_pure
]
for data, name in zip(data_list, name_list):
    try:
        # Perform prediction
        prediction = hls_model.predict(data)
        
        # Ensure prediction is a numpy array
        prediction = np.array(prediction)
        
        # Check if the prediction has the expected shape (n, 3)
        if prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}. Expected (n, 3)")
        
        # Write prediction to file
        output_filename = f"raw_predictions/{name}_prediction.dat"
        with open(output_filename, 'w') as f:
            for row in prediction:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")
        
        print(f"Prediction for {name} written to {output_filename}")
    
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")

print("All predictions completed.")

# iompare predictions
def load_dat_to_numpy(filename):
    # Read the file and split into lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Convert each line to a list of floats
    data = [list(map(float, line.strip().split(' '))) for line in lines]

    # Convert the list of lists to a NumPy array
    return np.array(data)

# Load the data
test_data = load_dat_to_numpy('tb_input.dat')
#test_data = Topo_2A_L1failed[:10000]
keras_predictions = model.predict(test_data)
hls_predictions = hls_model.predict(test_data)
with h5py.File('predictions.h5', 'w') as h5f:
    h5f.create_dataset('input', data=test_data)
    # Save the HLS predictions
    h5f.create_dataset('hls_predictions', data=hls_predictions)
    # Save the Keras predictions
    h5f.create_dataset('keras_predictions', data=keras_predictions)

#mse = np.mean((keras_predictions - hls_predictions)**2)
#print(f"Mean Squared Error between Keras and hls4ml predictions: {mse}")
print("original score")
print(keras_predictions)
print("new score")
print(hls_predictions)
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_L1failed)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_A14_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_HAHMggf_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_qqa_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_Zprime_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_ZZ4lep_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_HHbbtt_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_jz1_pure)
plt.show()
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_jz2_pure)
plt.show()
'''
import numpy as np

def compare_arrays_with_threshold(arr1, arr2, threshold):
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")

    # Calculate the absolute difference
    diff = np.abs(arr1 - arr2)

    # Count entries where difference is larger than threshold
    count_above_threshold = np.sum(diff > threshold)

    # Calculate percentage
    total_entries = arr1.size
    percentage = (count_above_threshold / total_entries) * 100

    return percentage
threshold = 0.1 
result = compare_arrays_with_threshold(keras_predictions, hls_predictions, threshold)
print(f"Percentage of entries with difference > {threshold}: {result:.5f}%")

##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_test_br)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_test_signal)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_HLT_passed)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_L1_passed)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_HHbbtt_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_jz1_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_jz2_pure)
plt.show()
##########################################################################################
#hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_jz4)
#plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_A14_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_HAHMggf_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_qqa_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_Zprime_pure)
plt.show()
##########################################################################################
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=Topo_2A_ZZ4lep_pure)
plt.show()
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def compare_anomaly_scores(signal_name, keras_predictions, hls_predictions):
    keras_scores = AD_score_CKL(keras_predictions)
    hls_scores = AD_score_CKL(hls_predictions)

    diff = abs(keras_scores-hls_scores)
    x_min = min(keras_scores.min(), hls_scores.min())
    x_min = min(x_min, diff.min())
    x_max = max(keras_scores.max(), hls_scores.max())

    plt.figure(figsize=(12, 6))
    sns.kdeplot(keras_scores, label='Pre-synthesis (Keras)', shade=True)
    sns.kdeplot(hls_scores, label='Post-synthesis (HLS)', shade=True)
    sns.kdeplot(abs(keras_scores-hls_scores), label='Pre-post difference', shade=True)

    plt.title(f'Anomaly Score Distribution Comparison for {signal_name}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min-x_max/20, x_max+x_max/20)
    plt.tight_layout()
    plt.savefig(f'{signal_name}_difference_FDL.png')
    plt.close()

signals = [
    Topo_2A_L1failed,
    Topo_2A_HHbbtt_pure,
    Topo_2A_jz1_pure,
    Topo_2A_jz2_pure,
    Topo_2A_A14_pure,
    Topo_2A_HAHMggf_pure,
    Topo_2A_qqa_pure,
    Topo_2A_Zprime_pure,
    Topo_2A_ZZ4lep_pure
]

signal_names = [
    "Topo_2A_test_signal",
    "Topo_2A_HLT_passed",
    "Topo_2A_L1_passed",
    "Topo_2A_HHbbtt_pure",
    "Topo_2A_jz1_pure",
    "Topo_2A_jz2_pure",
    "Topo_2A_jz4",
    "Topo_2A_A14_pure",
    "Topo_2A_HAHMggf_pure",
    "Topo_2A_qqa_pure",
    "Topo_2A_Zprime_pure",
    "Topo_2A_ZZ4lep_pure"
]

for signal, signal_name in zip(signals, signal_names):
    keras_predictions = model.predict(signal)
    hls_predictions = hls_model.predict(signal)
    
    compare_anomaly_scores(signal_name, keras_predictions, hls_predictions)
#############################################################################
def plot_all_ad_scores(signals, signal_names, hls_model):
    plt.figure(figsize=(15, 10))

    all_scores = []
    for signal, signal_name in zip(signals, signal_names):
        if 'jz4' not in signal_name.lower():
            hls_predictions = hls_model.predict(signal)
            ad_scores = AD_score_CKL(hls_predictions)
            all_scores.extend(ad_scores)
    
    # Determine global range for consistent binning
    min_score, max_score = min(all_scores), max(all_scores)
    bins = np.linspace(min_score, max_score, 16)  # 31 edges for 30 bins
    
    for signal, signal_name in zip(signals, signal_names):
        if 'jz4' not in signal_name.lower():
            hls_predictions = hls_model.predict(signal)
            ad_scores = AD_score_CKL(hls_predictions)
            
            # Create histogram
            hist, _ = np.histogram(ad_scores, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Normalize
            hist_normalized = hist / hist.max()
            
            if signal_name == 'Topo_2A_test_signal':

                plt.semilogy(bin_centers, hist_normalized, color='black', label=signal_name,

                             drawstyle='steps-mid', linewidth=2)

                plt.fill_between(bin_centers, hist_normalized, step='mid', alpha=0.3,

                                 color='lightgrey')

            else:

                plt.semilogy(bin_centers, hist_normalized, label=signal_name,

                             drawstyle='steps-mid', alpha=0.7)

    plt.title('Normalized AD Score Distribution Comparison (HLS predictions)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Normalized Counts (log scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig('all_processes_ad_score_distribution_log_30bins.png', dpi=300, bbox_inches='tight')
    plt.close()    
from scipy.stats import gaussian_kde
def plot_all_ad_scores_kde(signals, signal_names, hls_model):
    plt.figure(figsize=(15, 10))
    
    x_range = np.linspace(0, 0.6, 250)  
    
    for signal, signal_name in zip(signals, signal_names):
        if 'jz4' not in signal_name.lower():
            hls_predictions = hls_model.predict(signal)
            ad_scores = AD_score_CKL(hls_predictions)
            
            # Filter scores within the range 0 to 0.6
            ad_scores_filtered = ad_scores[(ad_scores >= 0) & (ad_scores <= 0.6)]
            
            # Compute KDE
            kde = gaussian_kde(ad_scores_filtered)
            y_values = kde(x_range)
            
            # Normalize
            y_normalized = y_values / y_values.max()
            
            # Plot
            if signal_name == 'Topo_2A_test_signal':
                plt.semilogy(x_range, y_normalized, color='black', label=signal_name, 
                         linewidth=2, marker='o', markersize=4)
                plt.fill_between(x_range, y_normalized, color='lightgrey', alpha=0.3)
            else:
                plt.semilogy(x_range, y_normalized, label=signal_name, alpha=0.7, 
                         marker='o', markersize=3)
                plt.fill_between(x_range, y_normalized, alpha=0.3)
    plt.title('Normalized AD Score Distribution Comparison (HLS predictions)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Normalized Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-6, 1)
    plt.tight_layout()
    plt.savefig('all_processes_ad_score_distribution_kde_0to0.6_30points.png', dpi=600, bbox_inches='tight')
    plt.close()
'''

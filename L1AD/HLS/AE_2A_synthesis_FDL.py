import hls4ml
import matplotlib.pyplot as plt
import plotting
from scipy.special import softmax, expit as sigmoid
from sklearn import metrics as sk
from keras.models import load_model
import os
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
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
from qkeras import quantized_relu, quantized_bits
print(tf.__version__)
print(keras.__version__)
print(tfmot.__version__)

#os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
os.environ['PATH'] = os.environ['XILINX_AP_INCLUDE'] + '/bin:' + os.environ['PATH']

model_path = '2A_AE_model_FDL_BESTOFLONGRUN'
custom_objects = {
    'QDense': QDense,
    'QActivation': QActivation,
    'QBatchNormalization': QBatchNormalization
}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
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
    if layer == "BN1":
        hls_config['LayerName'][layer]['Precision']['scale'] = 'ap_fixed<15,6>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<8,0>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<32,7>'
    #if layer == 'relu1':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == 'dense2':
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<5,1>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<22,3>'
    #if layer == 'dense2_linear':
        #hls_config['LayerName'][layer]['Precision']['table'] = 'ap_fixed<32,16>'
    if layer == "BN2":
        hls_config['LayerName'][layer]['Precision']['scale'] = 'ap_fixed<15,6>'
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<8,0>'
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<26,4>'
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
cfg['OutputDir'] = 'FiorDiLatte_Topo2A_trigger_VAE'
cfg['ClockPeriod'] = 25
cfg['ClockUncertainty'] = '27%'
cfg['Part'] = 'xcvu9p-flga2104-2-e'
cfg['ProjectName'] = 'Topo2A_AD_proj'

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()
hls_model.build(csim=False)


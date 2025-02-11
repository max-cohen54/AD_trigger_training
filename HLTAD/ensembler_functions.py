import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
import json
import matplotlib.pyplot as plt
import tf2onnx
import onnx
import onnxruntime as rt
import ROOT
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------------------
def load_subdicts_from_h5(save_dir):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5"):
            sub_dict_name = os.path.splitext(filename)[0]
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def combine_data(datasets, tags_to_combine, new_tag):
    """
    Combines subdicts of the 'datasets' dict.
    
    Inputs: 
        datasets: dict that maps {dataset_tag : dataset_dict}
        tags_to_combine: list if strings [dataset_tag1, ..., dataset_tagN] of the tags to be combined
        new_tag: the name of the new tag of the combined subdict

    Returns: 
        datasets: same datasets dict as input, but with the specified tags combined.
    """

    # initialize empty lists for new tag
    datasets[new_tag] = {key: [] for key in datasets[tags_to_combine[0]].keys()}

    # Loop through old tags and append np arrays to lists
    for tag in tags_to_combine:
        for key, value in datasets[tag].items():
            datasets[new_tag][key].append(value)

    # Concatenate lists into single np array
    for key, value in datasets[new_tag].items():
        datasets[new_tag][key] = np.concatenate(value, axis=0)

    # Delete old tags
    for tag in tags_to_combine:
        del datasets[tag]

    # Make sure everything is an np array
    for tag, data_dict in datasets.items():
        for key, value in data_dict.items():
            data_dict[key] = np.array(value)

    return datasets
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
class ZeroAwareStandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        # Initialize means and stds arrays with same shape as features
        n_features = X.shape[1]
        self.means = np.zeros(n_features)
        self.stds = np.zeros(n_features)

        # Compute mean and std for each feature, ignoring zero values
        for i in range(n_features):
            non_zero_values = X[:, i][X[:, i] != 0]
            if len(non_zero_values) > 0:
                self.means[i] = np.mean(non_zero_values)
                self.stds[i] = np.std(non_zero_values)
            else:
                self.means[i] = 0
                self.stds[i] = 1  # Avoid division by zero for empty or all-zero columns

    def transform(self, X):
        # Copy X to avoid changing the original dataset
        X_scaled = np.copy(X)

        # Apply scaling only to non-zero values
        for i in range(X.shape[1]):
            # Select non-zero values for feature i
            non_zero_mask = X[:, i] != 0
            X_scaled[non_zero_mask, i] = (X[non_zero_mask, i] - self.means[i]) / self.stds[i]

        return X_scaled
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def dphi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi # accounts for periodicity of phi
# -----------------------------------------------------------------------------------------


def find_threshold(scores, weights, pass_current_trigs, incoming_rate, target_rate):

    
    # Sort the inputs according to decreasing AD scores
    sorted_idxs = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idxs]
    sorted_weights = weights[sorted_idxs]
    sorted_pass_current_trigs = pass_current_trigs[sorted_idxs]

    # Calculate true and pure rates for each possible threshold
    mask = sorted_pass_current_trigs==False # Only consider events that fail the current HLT menu for pure rate
    cumulative_pure_rates = np.cumsum(sorted_weights * mask) * incoming_rate / np.sum(weights)
    cumulative_rates = np.cumsum(sorted_weights) * incoming_rate / np.sum(weights)

    # Select the threshold corresponding to the target rate
    threshold_idx = np.sum(cumulative_pure_rates < target_rate) - 1 # Since it is sorted in descending order
    threshold = sorted_scores[threshold_idx]

    # Calculate the pure and total rates corresponding to the selected threshold
    pass_AD_mask = sorted_scores >= threshold
    pure_rate = cumulative_pure_rates[pass_AD_mask][-1]
    total_rate = cumulative_rates[pass_AD_mask][-1]

    return threshold, pure_rate, total_rate




# -----------------------------------------------------------------------------------------
def load_and_preprocess(train_data_scheme: str, pt_normalization_type=None, L1AD_rate=1000, pt_thresholds=[50, 0, 0, 0], pt_scale_factor=0.05, comments=None):
    """
    Loads and preprocesses the training and signal data.

    Inputs:
        train_data_scheme: one of the strings defined below. Defines what data is used for training.
        pt_normalization_type: None, or one of the strings defined below. Defines how pt normalization is done.
        L1AD_rate: the pure rate at which the L1AD algo operates at. Used to calculate which events are L1Seeded.
        pt_thresholds: thresholds for which to zero out objects with pt below that value. [jet_threshold, electron_threshold, muon_threshold, photon_threshold].
        Comments: None or string. Any comments about the data / run that are worth noting down in the training documentation file.

    Returns:
        datasets: dict mapping {dataset_tag : dataset_dict} where dataset_dict is a sub-dict containing the data corresponding to that tag.
        data_info: dict containing information about the training data.
    """
    
    # Check arguments
    allowed_train_data_schemes = ['topo2A_train', 'L1noalg_HLTall', 'topo2A_train+L1noalg_HLTall']
    allowed_norm_types = ['per_event', 'global_division', 'StandardScaler', 'ZeroAwareStandardScaler']
    if (train_data_scheme not in allowed_train_data_schemes):
        raise ValueError(f"Invalid input: train_data_scheme {train_data_scheme}. Must be either None, or one of {allowed_train_data_schemes}")
    if (pt_normalization_type is not None) and (pt_normalization_type not in allowed_norm_types):
        raise ValueError(f"Invalid input: pt_normalization_type {pt_normalization_type}. Must be one of {allowed_norm_types}")

    # -------------------

    # Load data
    datasets = load_subdicts_from_h5('/eos/home-m/mmcohen/ad_trigger_development/data/loaded_and_matched_ntuples/02-03-2025')

    # put in Kaito's fake data:
    # First event data
    event1_jets = [
        [166.319, 0.034, 0.506], [80.189, -0.987, -1.825], [74.505, 2.859, 2.486],
        [45.475, -1.833, -2.604], [32.059, -0.992, 0.339], [28.418, -0.021, -0.379],
        [19.488, -2.061, 3.100], [18.555, 0.458, 1.031], [18.160, -1.457, -0.601],
        [15.961, 1.044, -1.224], [13.110, 0.912, 0.041], [12.259, -0.073, 2.738]
    ]
    event1_electrons = [
        [31.109, 0.038, 0.446], [31.109, 0.038, 0.446], [31.109, 0.038, 0.446],
        [18.804, 0.093, 0.500], [18.804, 0.093, 0.500], [18.804, 0.093, 0.500],
        [10.456, -1.817, -2.613], [8.816, -0.834, -1.980]
    ]
    event1_muons = []  # No muons
    event1_photons = [[11.828, -1.056, -1.836]]
    event1_met = [61.036, 0, -2.830]

    # Second event data
    event2_jets = [
        [98.170, -0.765, 2.707], [49.267, -0.119, 0.085], [21.128, 0.891, -1.510],
        [19.011, -1.372, 1.168], [16.770, -0.210, 1.878], [16.590, -1.377, 2.599],
        [14.321, 0.367, -0.439], [12.261, -1.167, -2.707], [11.575, 1.938, -1.190]
    ]
    event2_electrons = [[64.586, -0.723, 2.695]]
    event2_muons = []  # No muons
    event2_photons = [[62.634, -0.756, 2.699]]
    event2_met = [37.874, 0.0, -0.663]

    # Function to select the top-N pt objects or pad with zeros
    def select_objects(objects, max_count):
        if len(objects) > max_count:
            objects = sorted(objects, key=lambda x: -x[0])[:max_count]  # Top-N by pt
        while len(objects) < max_count:
            objects.append([0.0, 0.0, 0.0])  # Pad with zeros
        return objects

    # Process each event
    def process_event(jets, electrons, muons, photons, met):
        jets = select_objects(jets, 6)
        electrons = select_objects(electrons, 3)
        muons = select_objects(muons, 3)
        photons = select_objects(photons, 3)

        # Combine into a single array
        event_array = np.zeros((16, 3), dtype=np.float32)
        event_array[:6, :] = jets
        event_array[6:9, :] = electrons
        event_array[9:12, :] = muons
        event_array[12:15, :] = photons
        event_array[15, :] = met  # Last slot for MET
        return event_array

    # Create arrays for both events
    event1_array = process_event(event1_jets, event1_electrons, event1_muons, event1_photons, event1_met)
    event2_array = process_event(event2_jets, event2_electrons, event2_muons, event2_photons, event2_met)

    # Combine into a single (2, 16, 3) array
    combined_array = np.stack([event1_array, event2_array])
    
    datasets['kaito'] = {key: value[0:2] for key, value in datasets['topo2A_train'].items()}
    datasets['kaito']['HLT_data'] = combined_array
    datasets['kaito']['L1_data'] = combined_array

    #remove mc23e for some preliminary testing (to remove later):
    tags_to_remove = []
    for tag in datasets.keys():
        if ('qqa' in tag) or (tag=='jjJZ2') or (tag=='jjJZ4') or ('Zprime' in tag) or ('ZZ' in tag) or ('A14' in tag) or ('HHbbtt' in tag) or ('HAHM' in tag):
            tags_to_remove.append(tag)

    for tag in tags_to_remove:
        del datasets[tag]
    
   # -------------------
        
    # initialize training scheme
    if train_data_scheme == 'topo2A_train+L1noalg_HLTall':
        datasets = combine_data(datasets, tags_to_combine=['topo2A_train', 'HLT_noalg_eb_L1All'], new_tag='EB_train')

    elif train_data_scheme == 'L1noalg_HLTall':
        datasets['EB_train'] = datasets.pop('HLT_noalg_eb_L1All')

    elif train_data_scheme == 'topo2A_train':
        datasets['EB_train'] = datasets.pop('topo2A_train')

    # Only train over L1Passed events:
    passL1_mask = datasets['EB_train']['passL1'] ###########################
    datasets['EB_train'] = {key:value[passL1_mask] for key, value in datasets['EB_train'].items()} ###########################

    # now combine the other EB runs into EB_test
    tags_to_combine = [key for key in datasets.keys() if "EB" in key and key != 'EB_train']
    datasets = combine_data(datasets, tags_to_combine=tags_to_combine, new_tag='EB_test')
    
    # ------------------- 
    
    # # save raw data
    # for tag, data_dict in datasets.items():
    #     datasets[tag]['raw_HLT_data'] = np.copy(data_dict['HLT_data'])
    #     datasets[tag]['raw_L1_data'] = np.copy(data_dict['L1_data'])
    # # -------------------

    # Flatten ndarrays for use in DNN
    for tag, dict in datasets.items():
        for label, data in dict.items():
            if label.endswith('data'):
                datasets[tag][label] = np.reshape(data, newshape=(-1, 48))

    # -------------------

    # Split the train data into train + val
    indices = np.arange(len(datasets['EB_train']['HLT_data']))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=0)
    
    datasets['EB_val'] = {key:value[val_indices] for key, value in datasets['EB_train'].items()}
    datasets['EB_train'] = {key:value[train_indices] for key, value in datasets['EB_train'].items()}

    # -------------------

    data_info = {
        'train_data_scheme': train_data_scheme,
        'pt_normalization_type': pt_normalization_type,
        'L1AD_rate': L1AD_rate,
        'pt_thresholds': pt_thresholds,
        'pt_scale_factor': pt_scale_factor
    }
    if comments is not None:
        data_info['comments'] = comments

    return datasets, data_info



# -----------------------------------------------------------------------------------------
class DeltaPhiPreprocessingLayer(tf.keras.layers.Layer):
    def call(self, data):
        phi = data[:, :, 2]
        pt = data[:, :, 0]

        leading_jet_phi = phi[:, 0]
        pi = tf.constant(np.pi, dtype=tf.float32)  # Fix for tf.pi
        dphi = tf.math.mod(phi - tf.expand_dims(leading_jet_phi, axis=-1) + pi, 2 * pi) - pi

        zeroed_mask = tf.equal(pt, 0)
        phi_transformed = tf.where(zeroed_mask, tf.zeros_like(dphi), dphi)

        data_transformed = tf.concat([data[:, :, :2], tf.expand_dims(phi_transformed, axis=-1)], axis=-1)
        return data_transformed

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape


class METBiasMaskLayer(tf.keras.layers.Layer):
    def call(self, data):
        MET_values = data[:, -1, :]

        MET_zeros = tf.equal(MET_values[:, 0], 0)
        MET_neg999 = tf.equal(MET_values[:, 0], -999)
        MET_nan = tf.math.is_nan(MET_values[:, 2])

        MET_values = tf.where(tf.expand_dims(MET_zeros, axis=-1), tf.constant([[0.001, 0, 0]], dtype=data.dtype), MET_values)
        MET_values = tf.where(tf.expand_dims(MET_neg999 | MET_nan, axis=-1), tf.zeros_like(MET_values), MET_values)

        data_transformed = tf.concat([data[:, :-1, :], tf.expand_dims(MET_values, axis=1)], axis=1)
        return data_transformed



class ZeroOutLowPtLayer(tf.keras.layers.Layer):
    def __init__(self, pt_thresholds, **kwargs):
        super().__init__(**kwargs)
        self.pt_thresholds = pt_thresholds

    def call(self, data):
        jet_mask = tf.expand_dims(data[:, :6, 0] < self.pt_thresholds[0], axis=-1)
        electron_mask = tf.expand_dims(data[:, 6:9, 0] < self.pt_thresholds[1], axis=-1)
        muon_mask = tf.expand_dims(data[:, 9:12, 0] < self.pt_thresholds[2], axis=-1)
        photon_mask = tf.expand_dims(data[:, 12:15, 0] < self.pt_thresholds[3], axis=-1)

        data = tf.concat([
            tf.where(jet_mask, tf.zeros_like(data[:, :6, :]), data[:, :6, :]),
            tf.where(electron_mask, tf.zeros_like(data[:, 6:9, :]), data[:, 6:9, :]),
            tf.where(muon_mask, tf.zeros_like(data[:, 9:12, :]), data[:, 9:12, :]),
            tf.where(photon_mask, tf.zeros_like(data[:, 12:15, :]), data[:, 12:15, :]),
            data[:, 15:, :]
        ], axis=1)
        return data



class NormalizePtLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, data):
        data_scaled = tf.concat([
            tf.expand_dims(data[:, :, 0] * self.scale_factor, axis=-1),
            data[:, :, 1:]
        ], axis=-1)
        return data_scaled


class MSEADScoreLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        y, x = inputs
        mask = tf.logical_and(tf.not_equal(y, 0), tf.not_equal(y, -999))
        mask = tf.cast(mask, dtype=tf.float32)
        _y = y * mask
        _x = x * mask
        squared_diff = tf.square(_y - _x)
        sum_squared_diff = tf.reduce_sum(squared_diff, axis=-1)
        valid_count = tf.reduce_sum(mask, axis=-1)
        valid_count = tf.where(valid_count == 0, 1.0, valid_count)
        mse = sum_squared_diff / valid_count
        return mse



def create_large_AE_with_preprocessed_inputs(
    num_objects, num_features, h_dim_1, h_dim_2, h_dim_3, h_dim_4, latent_dim, 
    pt_thresholds, scale_factor, l2_reg=0.01, dropout_rate=0
):
    # Preprocessing Layers
    phi_rotation_layer = DeltaPhiPreprocessingLayer()
    met_bias_layer = METBiasMaskLayer()
    zero_out_layer = ZeroOutLowPtLayer(pt_thresholds)
    normalize_pt_layer = NormalizePtLayer(scale_factor)
    flatten_layer = tf.keras.layers.Flatten()

    # Preprocessing Model
    preprocessing_inputs = layers.Input(shape=(num_objects * num_features,))
    unflattened = tf.keras.layers.Reshape((num_objects, num_features))(preprocessing_inputs)
    preprocessed = phi_rotation_layer(unflattened)
    preprocessed = met_bias_layer(preprocessed)
    preprocessed = zero_out_layer(preprocessed)
    preprocessed = normalize_pt_layer(preprocessed)
    preprocessed_flattened = flatten_layer(preprocessed)

    preprocessing_model = tf.keras.Model(inputs=preprocessing_inputs, outputs=preprocessed_flattened)

    # Encoder (takes preprocessed input)
    encoder_inputs = layers.Input(shape=(num_objects * num_features,))  # Preprocessed input
    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    z = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)

    encoder = tf.keras.Model(inputs=encoder_inputs, outputs=z)

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_objects * num_features, kernel_regularizer=regularizers.l2(l2_reg))(x)
    decoder = tf.keras.Model(inputs=decoder_inputs, outputs=outputs)

    # Autoencoder (works directly on preprocessed input)
    ae_inputs = layers.Input(shape=(num_objects * num_features,))  # Preprocessed input
    reconstructed = decoder(encoder(ae_inputs))  # Encode and decode
    autoencoder = tf.keras.Model(inputs=ae_inputs, outputs=reconstructed)

    # MSE Model
    mse_scores = MSEADScoreLayer()([ae_inputs, reconstructed])  # Compare preprocessed input to reconstructed output
    mse_ae_model = tf.keras.Model(inputs=ae_inputs, outputs=mse_scores)

    return autoencoder, encoder, decoder, mse_ae_model, preprocessing_model

# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def loss_fn(y_true, y_pred):
    """Masked MSE with correct averaging by the number of valid objects."""
    
    # Masks to filter out invalid objects (zero padding and -999 placeholder)
    mask0 = K.cast(K.not_equal(y_true, 0), K.floatx())
    maskMET = K.cast(K.not_equal(y_true, -999), K.floatx())

    # Mask to upweight the first 6 elements (first two jets)
    weight = 1
    weight_mask = tf.ones_like(y_true)
    weight_mask = tf.concat([tf.ones_like(y_true[:, :6]) * weight, 
                      tf.ones_like(y_true[:, 6:])], 1)
    
    mask = mask0 * maskMET
    
    # Apply the mask to the squared differences
    squared_difference = K.square(mask * (y_pred - y_true)) * weight_mask
    
    # Sum the squared differences and the mask (to count valid objects)
    sum_squared_difference = K.sum(squared_difference, 1)
    valid_count = K.sum(mask, 1)  # Number of valid objects
    
    # Replace 0s by 1s
    valid_count = tf.where(K.equal(valid_count, 0), tf.ones_like(valid_count), valid_count)
    
    # Calculate the mean squared error by dividing by the number of valid objects
    mean_squared_error = sum_squared_difference / valid_count
    
    # Return the mean over the batch
    return K.mean(mean_squared_error)
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def initialize_model(input_dim, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, saved_model_path=None, save_version=None, obj_type='HLT'):
    '''
    Inputs:
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        saved_model_path: None or string. If string, loads the weights from the saved model path.
        save_version: None or string. If string, suffix of the model to be loaded.

    Returns:
        HLT_AE: full autoencoder model to be used with HLT objects
        HLT_encoder: just the encoder of HLT_AE
        L1_AE: full autoencoder model to be used with L1 objects
        L1_encoder: just the encoder of L1_AE
    '''

    # Initialize models
    INPUT_DIM = input_dim
    H_DIM_1 = 100
    H_DIM_2 = 100
    H_DIM_3 = 64
    H_DIM_4 = 32
    LATENT_DIM = latent_dim
        
    HLT_AE, HLT_encoder, HLT_decoder, HLT_MSE_AE, HLT_preprocessing_model = create_large_AE_with_preprocessed_inputs(
        num_objects=16, 
        num_features=3, 
        h_dim_1=H_DIM_1, 
        h_dim_2=H_DIM_2, 
        h_dim_3=H_DIM_3, 
        h_dim_4=H_DIM_4, 
        latent_dim=LATENT_DIM,
        pt_thresholds=pt_thresholds,
        scale_factor=pt_scale_factor,
        l2_reg=L2_reg_coupling, 
        dropout_rate=dropout_p
    )
    # -------------------

    # Compile
    optimizer = Adam(learning_rate=0.001)
    HLT_AE.compile(optimizer=optimizer, loss=loss_fn, weighted_metrics=[])
    # -------------------

    # Load model weights (if specified in the args)
    if (saved_model_path is None) != (save_version is None):
        raise ValueError("Either both or neither of 'saved_model_path' and 'save_version' should be None.")
        
    if (saved_model_path is not None) and (save_version is not None):
        HLT_AE.load_weights(f'{saved_model_path}/EB_{obj_type}_HLT_{save_version}.weights.h5')
        HLT_encoder.load_weights(f'{saved_model_path}/EB_{obj_type}_HLT_encoder_{save_version}.weights.h5')
        HLT_MSE_AE.load_weights(f'{saved_model_path}/EB_{obj_type}_MSE_HLT_AE_{save_version}.weights.h5')
        HLT_preprocessing_model.load_weights(f'{saved_model_path}/EB_{obj_type}_preprocessing_A{save_version}.weights.h5')
    # -------------------

    return HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model


# -----------------------------------------------------------------------------------------
def train_model(datasets: dict, model_version: str, save_path: str, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, training_weights=True, obj_type='HLT'):
    """
    Trains, and saves an AE.

    Inputs:
        datasets: dict containing the data.
        
    Returns: None
    """

    model_args = {
        'input_dim': datasets['EB_train']['HLT_data'].shape[1],
        'pt_thresholds': pt_thresholds,
        'pt_scale_factor': pt_scale_factor,
        'dropout_p': dropout_p,
        'L2_reg_coupling': L2_reg_coupling,
        'latent_dim': latent_dim,
        'obj_type': obj_type,
    }
    
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(**model_args)

    # Define callbacks
    STOP_PATIENCE = 9
    LR_PATIENCE = 6
    early_stopping = EarlyStopping(patience=STOP_PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=LR_PATIENCE, verbose=1)
    callbacks = [early_stopping, reduce_lr]
    # -------------------

    # Train and save models
    NUM_EPOCHS = 500
    BATCH_SIZE = 512

    if training_weights:
        train_weights = 1000*datasets['EB_train']['weights']/np.sum(datasets['EB_train']['weights'])
        val_weights = 1000*datasets['EB_val']['weights']/np.sum(datasets['EB_val']['weights'])
    else:
        train_weights = np.ones_like(datasets['EB_train']['weights'])
        val_weights = np.ones_like(datasets['EB_val']['weights'])

    # Calculate the preprocessed data
    datasets['EB_train'][f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(datasets['EB_train'][f'{obj_type}_data'], batch_size=8)
    datasets['EB_val'][f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(datasets['EB_val'][f'{obj_type}_data'], batch_size=8)
        
    history = HLT_AE.fit(
        x=datasets['EB_train'][f'{obj_type}_preprocessed_data'], 
        y=datasets['EB_train'][f'{obj_type}_preprocessed_data'], 
        validation_data=(datasets['EB_val'][f'{obj_type}_preprocessed_data'], datasets['EB_val'][f'{obj_type}_preprocessed_data'], val_weights),
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks, 
        sample_weight = train_weights,
        verbose=0
    )

    HLT_AE.save_weights(f'{save_path}/EB_{obj_type}_HLT_{model_version}.weights.h5')
    HLT_encoder.save_weights(f'{save_path}/EB_{obj_type}_HLT_encoder_{model_version}.weights.h5')
    HLT_MSE_AE.save_weights(f'{save_path}/EB_{obj_type}_MSE_HLT_AE_{model_version}.weights.h5')
    HLT_preprocessing_model.save_weights(f'{save_path}/EB_{obj_type}_preprocessing_A{model_version}.weights.h5')
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def train_multiple_models(datasets: dict, data_info: dict, save_path: str, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, num_trainings=10, training_weights=True, obj_type='HLT'):
    """
    calls 'initialize_and_train' multiple times to average results across multiple trainings.

    Inputs:
        datasets: dict containing the data.
        data_info: dict containing information about the training data.
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        num_trainings: number of trainings to run.

    Returns:
        training_info: dict containing information about the training which will be used for processing the model. Also dumped into a file for documentation.
        data_info: dict containing information about the training data.
    """
    
    # -------------------
    print(f'Booting up... initializing trainings of {num_trainings} models\n')

    for i in range(num_trainings):
        print(f'starting training model {i}...')
        
        model_version = f'{i}'
        
        train_model(
            datasets=datasets, 
            model_version=model_version,
            save_path=save_path,
            dropout_p=dropout_p,
            pt_thresholds=data_info['pt_thresholds'],
            pt_scale_factor=data_info['pt_scale_factor'],
            L2_reg_coupling=L2_reg_coupling,
            latent_dim=latent_dim,
            large_network=large_network,
            training_weights=training_weights,
            obj_type=obj_type
        )
        print(f'model {i} success\n')

    print(f'Powering off... finished trainings.')
    # -------------------

    # -------------------
    training_info = {
        'save_path': save_path,
        'dropout_p': dropout_p,
        'L2_reg_coupling': L2_reg_coupling,
        'latent_dim': latent_dim,
        'large_network': large_network,
        'num_trainings': num_trainings,
        'training_weights': training_weights,
        'obj_type': obj_type
    }
    # -------------------

    # Write the training info to a txt file
    with open('./training_documentation.txt', 'a') as f:
        f.write('\n training_info:')
        f.write(json.dumps(training_info))
        f.write('data_info:')
        f.write(json.dumps(data_info))
        f.write('\n')
    # -------------------

    return training_info, data_info
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def MSE_AD_score(y, x):
    '''
    Calculates MSE between element of y, x, skipping over missing objects

    Inputs:
        y: array of data (e.g. original samples)
        x: other array of data (e.g. model outputs)

    Returns:
        loss: Array of (masked) MSE values between y and x (e.g. loss[i] = MSE(y, x))
    '''
    # Create the mask where the valid data points are (not 0 or -999)
    mask = (y != 0) & (y != -999)
    
    _x = x * mask
    _y = y * mask
    
    squared_diff = np.square(_y - _x)
    sum_squared_diff = np.sum(squared_diff, axis=-1)
    
    # Count the number of valid (non-masked) elements
    valid_count = np.sum(mask, axis=-1)
    
    # Avoid division by zero by replacing 0 counts with 1
    valid_count = np.where(valid_count == 0, 1, valid_count)
    
    # Calculate the mean squared error for each event
    loss = sum_squared_diff / valid_count
    
    return loss
# -----------------------------------------------------------------------------------------

def ROC_curve_plot(datasets: dict, save_path: str, save_name: str, HLTAD_threshold, bkg_tag='EB_test', obj_type='HLT'):
    
    # Check if the background tag is valid
    if bkg_tag not in datasets.keys():
        raise ValueError(f"Invalid bkg_tag: {bkg_tag}. Must be in the keys of the datasets dictionary.")
    
    # Start the plot
    plt.figure(figsize=(15, 8))
    plt.rcParams['axes.linewidth'] = 2.4

    # Initialize the signal efficiencies dictionary
    signal_efficiencies = {}

    # Get the background scores and weights
    bkg_scores = datasets[bkg_tag][f'{obj_type}_AD_scores']
    bkg_weights = datasets[bkg_tag]['weights']
    
    # Loop over each tag
    skip_tags = ['EB_train', 'EB_val', bkg_tag]
    for tag, data_dict in datasets.items():
        if tag in skip_tags: continue

        # Get the signal scores and weights
        signal_scores = data_dict[f'{obj_type}_AD_scores']
        signal_weights = data_dict['weights']

        # Combine the background and signal
        combined_scores = np.concatenate((bkg_scores, signal_scores), axis=0)
        combined_weights = np.concatenate((bkg_weights, signal_weights), axis=0)
        combined_labels = np.concatenate((np.zeros_like(bkg_scores), np.ones_like(signal_scores)), axis=0)

        # Use sklearn to calculate the ROC curve
        FPRs, TPRs, thresholds = roc_curve(y_true=combined_labels, y_score=combined_scores, sample_weight=combined_weights)
        AUC = auc(FPRs, TPRs)

        # Calculate the TPR at target FPR
        #closest_index = np.argmin(np.abs(FPRs - target_FPR))
        closest_index = np.argmin(np.abs(thresholds - HLTAD_threshold))
        corresponding_FPR = FPRs[closest_index]
        corresponding_TPR = TPRs[closest_index]
        signal_efficiencies[tag] = corresponding_TPR
        print(f'ROC threshold for tag {tag} was given to be {thresholds[closest_index]}')
        print(f'Corresponding ROC FPR is {corresponding_FPR}')

        # Add the ROC curve from this tag to the plot
        plt.plot(FPRs, TPRs, label=f'{tag}, AUC={AUC:.3f}', linewidth=1.5)

    # Plot diagonal line
    xx = np.linspace(0, 1, 100)
    plt.plot(xx, xx, color='grey', linestyle='dashed')

    # Plot vertical line corresponding to 10Hz HLTAD rate
    plt.plot([corresponding_FPR, corresponding_FPR], [0, 1], color='r', linestyle='dashed')

    # Aesthetics
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(save_name, fontsize=14)
    plt.legend(fontsize=12, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return signal_efficiencies

def raw_efficiencies_plot_from_ROC(signal_efficiencies, save_path: str, save_name: str):
    # Start the plot

    y = np.arange(len(signal_efficiencies.keys()))

    # Plot for L1Seeded
    plt.figure(figsize=(15,8))
    plt.scatter(signal_efficiencies.values(), y, color='cornflowerblue', s=150, alpha=0.5)
    plt.xlabel('Efficiency', fontsize=15)
    plt.title(f'Raw Signal Efficiencies', fontsize=16)
    plt.yticks(y, signal_efficiencies.keys())
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    #plt.savefig(f'{save_path}/ROC_{model_version}_efficiencies.png')
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

def raw_efficiencies_plot_from_regions(region_counts: dict, save_path: str, save_name: str, seed_scheme: str):
    
    if seed_scheme not in ['l1Seeded', 'l1All']:
        raise ValueError(f"Invalid seed_scheme: {seed_scheme}. Must be 'l1Seeded' or 'l1All'.")

    # Calculate the efficiencies from the region counts
    if seed_scheme == 'l1Seeded':
        signal_efficiencies = {key: (counts['E'] + counts['F']) / (counts['B'] + counts['C']) for key, counts in region_counts.items() if (counts['B'] + counts['C']) != 0}

    elif seed_scheme == 'l1All':
        signal_efficiencies = {key: (counts['E'] + counts['F']) / (counts['B'] + counts['C'] + counts['D']) for key, counts in region_counts.items() if (counts['B'] + counts['C'] + counts['D']) != 0}

    y = np.arange(len(signal_efficiencies.keys()))

    # Plot
    plt.figure(figsize=(15,8))
    plt.scatter(signal_efficiencies.values(), y, color='cornflowerblue', s=150, alpha=0.5)
    plt.xlabel('Efficiency', fontsize=15)
    plt.title(f'Raw Signal Efficiencies', fontsize=16)
    plt.yticks(y, signal_efficiencies.keys())
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    #plt.savefig(f'{save_path}/region_counts_{model_version}_efficiencies.png')
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return signal_efficiencies

def efficiency_gain_plot(region_counts: dict, save_path: str, save_name: str, target_rate: int):

    tags = [tag for tag in region_counts.keys() if region_counts[tag]['A'] != 0 and not (tag.startswith('k') or tag.startswith('EB') or tag.startswith('phys') or ('noalg' in tag))]

    hlt_effs = [(region_counts[tag]['G'] + region_counts[tag]['F']) / (region_counts[tag]['A']) for tag in tags]
    combined_effs = [(region_counts[tag]['E'] + region_counts[tag]['G'] + region_counts[tag]['F']) / (region_counts[tag]['A']) for tag in tags]
    
    efficiency_gains = [(combined_effs - hlt_effs) * 100 / hlt_effs if hlt_effs != 0 else 999 for hlt_effs, combined_effs in zip(hlt_effs, combined_effs)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    y = np.arange(len(tags))

    ax1.scatter(hlt_effs, y, label='HLT Efficiency', color='cornflowerblue', s=150, alpha=0.5)
    ax1.scatter(combined_effs, y, label='HLT + AD Efficiency', color='mediumvioletred', s=150, alpha=0.5)

    ax1.set_xlabel('Efficiency', fontsize=15)
    ax1.set_title(f'Efficiency Comparison', fontsize=16)
    ax1.set_yticks(y)
    ax1.set_yticklabels(tags)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.legend()

    ax2.barh(y, efficiency_gains, color='seagreen', edgecolor='k')
    for i, gain in enumerate(efficiency_gains):
        ax2.text(gain + 0.5, y[i], f'{gain:.4f}%', va='center', color='black')

    # Set x axis limit -------------------------------------------
    # Convert to a NumPy array for easier manipulation
    valid_efficiency_gains = np.array(efficiency_gains)
    
    if valid_efficiency_gains.size > 0:
        # Check if 999 exists and find the next-highest value
        if 999 in valid_efficiency_gains:
            max_limit = np.sort(valid_efficiency_gains[valid_efficiency_gains < 999])[-1] + 2
        else:
            max_limit = np.max(valid_efficiency_gains) + 2
    
        ax2.set_xlim(0, max_limit)
    else:
        ax2.set_xlim(0, 2)  # Fallback if no valid values exist
    # -------------------------------------------------------------

    ax2.set_xlabel('Efficiency Gain (%)', fontsize=15)
    ax2.set_title('Relative Efficiency Gain', fontsize=15)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
    

def EoverFplusG_plot(region_counts: dict, save_path: str, save_name: str):

    EoverFplusG = {key: (counts['E']) / (counts['F'] + counts['G']) for key, counts in region_counts.items() if (counts['F'] + counts['G']) != 0}

    y = np.arange(len(EoverFplusG.keys()))

    # Plot for L1Seeded
    plt.figure(figsize=(15,8))
    plt.scatter(EoverFplusG.values(), y, color='cornflowerblue', s=150, alpha=0.5)
    plt.xlabel('Efficiency', fontsize=15)
    plt.title(f'Signal Efficiency Gains (E/(F+G))', fontsize=16)
    plt.yticks(y, EoverFplusG.keys())
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return EoverFplusG

def efficiency_vs_variable_plot(datasets: dict, save_path: str, save_name: str, obj_type: str, seed_scheme: str):

    # idxs = ['B' in label for label in datasets['EB_test']['region_labels']] | ['C' in label for label in datasets['EB_test']['region_labels']]
    # anomalous = (['E' in label for label in datasets['EB_test']['region_labels']] | ['F' in label for label in datasets['EB_test']['region_labels']])

    pileups = datasets['EB_test']['pileups']
    leading_jet_pt = datasets['EB_test'][f'{obj_type}_preprocessed_data'][:, 0]
    MET_pt = datasets['EB_test'][f'{obj_type}_preprocessed_data'][:, -3]
    weights = datasets['EB_test']['weights']

    # Define bins for each variable
    pileup_bins = np.linspace(np.min(pileups[pileups != 0])-5, np.max(pileups)+5, 35)
    jet_pt_bins = np.linspace(np.min(leading_jet_pt)-5, np.percentile(leading_jet_pt, 75)+300, 35)
    MET_pt_bins = np.linspace(np.min(MET_pt)-5, np.percentile(MET_pt, 75)+300, 35)

    # Initialize TEfficiency for pileup
    h_total_pileup = ROOT.TH1F("h_total_pileup", "Total Events Pileup", len(pileup_bins)-1, pileup_bins)
    h_pass_pileup = ROOT.TH1F("h_pass_pileup", "Passed Events Pileup", len(pileup_bins)-1, pileup_bins)

    # Initialize TEfficiency for leading jet pt
    h_total_jet_pt = ROOT.TH1F("h_total_jet_pt", "Total Events Jet Pt", len(jet_pt_bins)-1, jet_pt_bins)
    h_pass_jet_pt = ROOT.TH1F("h_pass_jet_pt", "Passed Events Jet Pt", len(jet_pt_bins)-1, jet_pt_bins)

    # Initialize TEfficiency for MET pt
    h_total_MET_pt = ROOT.TH1F("h_total_MET_pt", "Total Events MET Pt", len(MET_pt_bins)-1, MET_pt_bins)
    h_pass_MET_pt = ROOT.TH1F("h_pass_MET_pt", "Passed Events MET Pt", len(MET_pt_bins)-1, MET_pt_bins)

    if seed_scheme == 'l1Seeded':
        L1Seeded_mask = np.array(['B' in label for label in datasets['EB_test']['region_labels']]) | np.array(['C' in label for label in datasets['EB_test']['region_labels']])
        HLTAD_mask = np.array(['E' in label for label in datasets['EB_test']['region_labels']]) | np.array(['F' in label for label in datasets['EB_test']['region_labels']])

    elif seed_scheme == 'l1All':
        L1Seeded_mask = np.array(['B' in label for label in datasets['EB_test']['region_labels']]) | np.array(['C' in label for label in datasets['EB_test']['region_labels']]) | np.array(['D' in label for label in datasets['EB_test']['region_labels']])
        HLTAD_mask = np.array(['E' in label for label in datasets['EB_test']['region_labels']]) | np.array(['F' in label for label in datasets['EB_test']['region_labels']])

    # Fill histograms using pileup, leading jet pt, MET pt, and weights
    for i in range(len(datasets['EB_test']['pileups'])):

        # Fill the total histograms with events passing L1AD (regions B and C)
        if L1Seeded_mask[i]:
            h_total_pileup.Fill(datasets['EB_test']['pileups'][i], datasets['EB_test']['weights'][i])
            h_total_jet_pt.Fill(leading_jet_pt[i], datasets['EB_test']['weights'][i])
            h_total_MET_pt.Fill(MET_pt[i], datasets['EB_test']['weights'][i])
        
        # Fill the pass histograms with events passing HLTAD (regions E and F)
        if HLTAD_mask[i]:
            h_pass_pileup.Fill(datasets['EB_test']['pileups'][i], datasets['EB_test']['weights'][i])
            h_pass_jet_pt.Fill(leading_jet_pt[i], datasets['EB_test']['weights'][i])
            h_pass_MET_pt.Fill(MET_pt[i], datasets['EB_test']['weights'][i])

    # Create TEfficiency objects
    eff_pileup = ROOT.TEfficiency(h_pass_pileup, h_total_pileup)
    eff_jet_pt = ROOT.TEfficiency(h_pass_jet_pt, h_total_jet_pt)
    eff_MET_pt = ROOT.TEfficiency(h_pass_MET_pt, h_total_MET_pt)

    # Plot efficiency vs pileup using ROOT
    if save_path is not None:
        c_pileup = ROOT.TCanvas("c_pileup", "Efficiency vs Pileup", 800, 600)
        eff_pileup.SetTitle(f"Anomalous Event Efficiency vs Pileup;Pileup;Efficiency")
        eff_pileup.Draw("AP")
        c_pileup.SaveAs(f'{save_path}/{save_name}_pileup.png')

        # Plot efficiency vs leading jet pt using ROOT
        c_jet_pt = ROOT.TCanvas("c_jet_pt", "Efficiency vs Leading Jet Pt", 800, 600)
        eff_jet_pt.SetTitle(f"Anomalous Event Efficiency vs Leading Jet Pt;Leading Jet Pt;Efficiency")
        eff_jet_pt.Draw("AP")
        c_jet_pt.SaveAs(f'{save_path}/{save_name}_jet_pt.png')

        # Plot efficiency vs MET pt using ROOT
        c_MET_pt = ROOT.TCanvas("c_MET_pt", "Efficiency vs MET Pt", 800, 600)
        eff_MET_pt.SetTitle(f"Anomalous Event Efficiency vs MET Pt;MET Pt;Efficiency")
        eff_MET_pt.Draw("AP")
        c_MET_pt.SaveAs(f'{save_path}/{save_name}_MET_pt.png')

def plot_individual_model_results(datasets: dict, region_counts: dict, seed_scheme, save_path, model_version, L1AD_threshold, L1AD_rate, HLTAD_threshold, target_HLTAD_rate, obj_type='HLT'):

    if seed_scheme not in ['l1Seeded', 'l1All']:
        raise ValueError(f"Invalid seed_scheme: {seed_scheme}. Must be 'l1Seeded' or 'l1All'.")
    
    if seed_scheme == 'l1Seeded':

        # Target FPR = HLT_trigger_rate / incoming_rate
        seeded_target_FPR = target_HLTAD_rate / L1AD_rate
        
        # Create a new dataset dict that only contains events that pass the L1AD threshold
        seeded_datasets = {}
        for tag in datasets.keys():
            pass_L1AD = datasets[tag]['topo2A_AD_scores'] >= L1AD_threshold
            seeded_datasets[tag] = {key: np.array(value)[pass_L1AD] for key, value in datasets[tag].items()}

    elif seed_scheme == 'l1All':
        # here, HLTAD sees all events passing L1AD or L1 (roughly 100kHz)
        seeded_target_FPR = target_HLTAD_rate / 100000

        # Create a new dataset dict that contains events passing L1AD or L1
        seeded_datasets = {}
        for tag in datasets.keys():
            pass_L1AD_or_L1 = (datasets[tag]['topo2A_AD_scores'] >= L1AD_threshold) | (datasets[tag]['passL1'])
            seeded_datasets[tag] = {key: np.array(value)[pass_L1AD_or_L1] for key, value in datasets[tag].items()}

    # Plot the ROC curves and obtain raw signal efficiencies
    seeded_signal_efficiencies = ROC_curve_plot(seeded_datasets, save_path=save_path, save_name=f'ROC_curves_{model_version}_{seed_scheme}', HLTAD_threshold=HLTAD_threshold, obj_type=obj_type)

    # Plot the raw efficiencies
    raw_efficiencies_plot_from_ROC(seeded_signal_efficiencies, save_path=save_path, save_name=f'Efficiencies_ROC_{model_version}_{seed_scheme}')
    signal_efficiencies = raw_efficiencies_plot_from_regions(region_counts, save_path=save_path, save_name=f'Efficiencies_region_counts_{model_version}_{seed_scheme}', seed_scheme=seed_scheme)

    # Plot the efficiency gains
    EoverFplusG = EoverFplusG_plot(region_counts, save_path=save_path, save_name=f'EoverFplusG_{model_version}_{seed_scheme}')
    efficiency_gain_plot(region_counts, save_path=save_path, save_name=f'Efficiency_gains_{model_version}_{seed_scheme}', target_rate=target_HLTAD_rate)

    # Plot the efficiency vs variable
    efficiency_vs_variable_plot(datasets, save_path=save_path, save_name=f'Efficiency_plot_{model_version}_{seed_scheme}', obj_type=obj_type, seed_scheme=seed_scheme)

    return signal_efficiencies, EoverFplusG


def ensemble_efficiency_gain_plot(efficiency_gains: dict, save_path: str, save_name: str):

    good_tags = [tag for tag in efficiency_gains[0].keys() if not (tag.startswith('k') or tag.startswith('EB') or tag.startswith('phys') or ('noalg' in tag))]
    
    gains = {tag: [gains[tag] for gains in efficiency_gains] for tag in good_tags}


    plt.figure(figsize=(12, 6))
    plt.boxplot(gains.values(), labels=gains.keys())
    plt.title(f'Distribution of Efficiency Gains')
    plt.ylabel('Efficiency Gain (E/(F+G))')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

def ensemble_raw_efficiencies_plot(efficiencies: dict, save_path: str, save_name: str):

    effs = {tag: [effs[tag] for effs in efficiencies] for tag in efficiencies[0].keys()}

    plt.figure(figsize=(12, 6))
    plt.boxplot(effs.values(), labels=effs.keys())
    plt.title(f'Distribution of Efficiencies')
    plt.ylabel('Efficiency (E/(F+G))')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

def plot_ensemble_results(results: dict, save_path: str, seed_scheme: str):
    efficiency_gains = results['efficiency_gains']
    efficiencies = results['efficiencies']

    ensemble_efficiency_gain_plot(efficiency_gains, save_path, save_name=f'Efficiency_gains_distribution_{seed_scheme}')
    ensemble_raw_efficiencies_plot(efficiencies, save_path, save_name=f'Efficiencies_distribution_{seed_scheme}')




    
# -----------------------------------------------------------------------------------------
def process_multiple_models(training_info: dict, data_info: dict, plots_path: str, target_rate: int=10, L1AD_rate: int=1000, custom_datasets=None, obj_type='HLT'):

    print(f'powering on... preparing to run evals')

    if custom_datasets is not None:
        datasets = custom_datasets
    else:
        # Load data according to the training info
        datasets, data_info = load_and_preprocess(**data_info)

    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    #large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    print(f'evals phase 1 of 2 initiated.')

    # Initialize results: dictionary of lists. Each list will contain 'num_trainings' elements.
    # trying to decide if I want lists for each metric, or if I should first separate it by tags, and then have metrics in each tag.
    l1Seeded_results = {
        'region_counts': [],
        'efficiency_gains': [],
        'efficiencies': []
    }

    l1All_results = {
        'region_counts': [],
        'efficiency_gains': [],
        'efficiencies': []
    }

    # Loop over each trained model
    for i in range(num_trainings):

        print(f'phase 1: starting evals of model {i}...')

        # Load the model
        HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
            input_dim=datasets['EB_train']['HLT_data'].shape[1],
            pt_thresholds=data_info['pt_thresholds'],
            pt_scale_factor=data_info['pt_scale_factor'],
            dropout_p=dropout_p,
            L2_reg_coupling=L2_reg_coupling,
            latent_dim=latent_dim,
            #large_network=large_network,
            saved_model_path=save_path,
            save_version=i,
            obj_type=obj_type
        )

        # Pass the data through the model
        skip_tags = ['EB_train', 'EB_val']
        for tag, dict in datasets.items():
            if tag in skip_tags: continue

            dict[f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(dict[f'{obj_type}_data'], verbose=0, batch_size=8)
            #dict[f'{obj_type}_model_outputs'] = HLT_AE.predict(dict[f'{obj_type}_preprocessed_data'], verbose=0, batch_size=8)
            #dict[f'{obj_type}_latent_reps'] = HLT_encoder.predict(dict[f'{obj_type}_preprocessed_data'], verbose=0, batch_size=8)

        # Calculate the AD scores
        for tag, dict in datasets.items():
            if tag in skip_tags: continue

            dict[f'{obj_type}_AD_scores'] = HLT_MSE_AE.predict(dict[f'{obj_type}_preprocessed_data'], batch_size=8)
            #dict[f'calculated_{obj_type}_AD_scores'] = MSE_AD_score(dict[f'{obj_type}_preprocessed_data'], dict[f'{obj_type}_model_outputs'])

        
        # Calculate the L1AD threshold and rates
        L1AD_threshold, L1AD_pure_rate , L1AD_total_rate = find_threshold(
            scores=datasets['EB_test']['topo2A_AD_scores'],
            weights=datasets['EB_test']['weights'],
            pass_current_trigs=datasets['EB_test']['passL1'],
            target_rate=L1AD_rate,
            incoming_rate=31575960
        )

        print(f'model {i}:')
        print(f'L1AD_pure_rate: {L1AD_pure_rate}')
        print(f'L1AD_total_rate: {L1AD_total_rate}')
        print(f'L1AD_threshold: {L1AD_threshold}')


        # L1Seeded ---------------------------------------------------------
        pass_L1AD_mask = datasets['EB_test']['topo2A_AD_scores'] >= L1AD_threshold

        HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
            scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
            weights=datasets['EB_test']['weights'][pass_L1AD_mask],
            pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
            target_rate=target_rate,
            incoming_rate=L1AD_total_rate
        )
        print(f'l1Seeded:::')
        print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
        print(f'HLTAD_total_rate: {HLTAD_total_rate}')
        print(f'HLTAD_threshold: {HLTAD_threshold}\n')

        # Initialize the region counts for each tag
        region_counts = {tag: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0} for tag in datasets.keys()}
        
        # Loop over each tag
        for tag, data_dict in datasets.items():
            if tag in skip_tags: continue
            
            # will hold the regions that each event falls into. Each event will be in region A
            data_dict['region_labels'] = ['A'] * len(data_dict[f'{obj_type}_AD_scores'])

            passHLTAD = data_dict[f'{obj_type}_AD_scores'] >= HLTAD_threshold
            passHLT = data_dict['passHLT']
            passL1AD = data_dict['topo2A_AD_scores'] >= L1AD_threshold
            passL1 = data_dict['passL1']

            # Add letters to strings where conditions are met
            for j, (l1ad, l1, hltad, hlt) in enumerate(zip(passL1AD, passL1, passHLTAD, passHLT)):
                if l1ad and not l1:
                    data_dict['region_labels'][j] += 'B'
                if l1ad and l1:
                    data_dict['region_labels'][j] += 'C'
                if not l1ad and l1:
                    data_dict['region_labels'][j] += 'D'
                if l1ad and hltad and not hlt:
                    data_dict['region_labels'][j] += 'E'
                if l1ad and hltad and hlt:
                    data_dict['region_labels'][j] += 'F'
                if (l1 or l1ad) and not hltad and hlt:
                    data_dict['region_labels'][j] += 'G'

            # Now keep track of the number of events in each region
            for j, label in enumerate(data_dict['region_labels']):
                weight = data_dict['weights'][j]
                for region in label:
                    region_counts[tag][region] += weight
        
        # Append the results to the list
        l1Seeded_results['region_counts'].append(region_counts)

        #target_FPR = L1Seeded_HLTAD_total_rate / L1AD_total_rate

        # Now let's make plots for this model
        signal_efficiencies, EoverFplusG = plot_individual_model_results(
            datasets=datasets, 
            region_counts=region_counts, 
            seed_scheme='l1Seeded',
            save_path=plots_path, 
            model_version=i, 
            L1AD_threshold=L1AD_threshold, 
            L1AD_rate=L1AD_total_rate, 
            HLTAD_threshold=HLTAD_threshold,
            target_HLTAD_rate=target_rate,
            obj_type=obj_type
        )

        l1Seeded_results['efficiency_gains'].append(EoverFplusG)
        l1Seeded_results['efficiencies'].append(signal_efficiencies)

        # L1All ---------------------------------------------------------
        pass_L1AD_or_L1_mask = (datasets['EB_test']['topo2A_AD_scores'] >= L1AD_threshold) | (datasets['EB_test']['passL1'])

        HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
            scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_or_L1_mask],
            weights=datasets['EB_test']['weights'][pass_L1AD_or_L1_mask],
            pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_or_L1_mask],
            target_rate=target_rate,
            incoming_rate=100000
        )
        print(f'l1All:::')
        print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
        print(f'HLTAD_total_rate: {HLTAD_total_rate}')
        print(f'HLTAD_threshold: {HLTAD_threshold}\n')

        # Initialize the region counts for each tag
        region_counts = {tag: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0} for tag in datasets.keys()}
        
        # Loop over each tag
        for tag, data_dict in datasets.items():
            if tag in skip_tags: continue
            
            # will hold the regions that each event falls into. Each event will be in region A
            data_dict['region_labels'] = ['A'] * len(data_dict[f'{obj_type}_AD_scores'])

            passHLTAD = data_dict[f'{obj_type}_AD_scores'] >= HLTAD_threshold
            passHLT = data_dict['passHLT']
            passL1AD = data_dict['topo2A_AD_scores'] >= L1AD_threshold
            passL1 = data_dict['passL1']

            # Add letters to strings where conditions are met
            for j, (l1ad, l1, hltad, hlt) in enumerate(zip(passL1AD, passL1, passHLTAD, passHLT)):
                if l1ad and not l1:
                    data_dict['region_labels'][j] += 'B'
                if l1ad and l1:
                    data_dict['region_labels'][j] += 'C'
                if not l1ad and l1:
                    data_dict['region_labels'][j] += 'D'
                if (l1ad or l1) and hltad and not hlt:
                    data_dict['region_labels'][j] += 'E'
                if (l1ad or l1) and hltad and hlt:
                    data_dict['region_labels'][j] += 'F'
                if (l1 or l1ad) and not hltad and hlt:
                    data_dict['region_labels'][j] += 'G'

            # Now keep track of the number of events in each region
            for j, label in enumerate(data_dict['region_labels']):
                weight = data_dict['weights'][j]
                for region in label:
                    region_counts[tag][region] += weight
        
        # Append the results to the list
        l1All_results['region_counts'].append(region_counts)

        #target_FPR = L1Seeded_HLTAD_total_rate / L1AD_total_rate

        # Now let's make plots for this model
        signal_efficiencies, EoverFplusG = plot_individual_model_results(
            datasets=datasets, 
            region_counts=region_counts, 
            seed_scheme='l1All',
            save_path=plots_path, 
            model_version=i, 
            L1AD_threshold=L1AD_threshold, 
            L1AD_rate=L1AD_total_rate, 
            HLTAD_threshold=HLTAD_threshold,
            target_HLTAD_rate=target_rate,
            obj_type=obj_type
        )

        l1All_results['efficiency_gains'].append(EoverFplusG)
        l1All_results['efficiencies'].append(signal_efficiencies)

    
    plot_ensemble_results(l1Seeded_results, save_path=plots_path, seed_scheme='l1Seeded')
    plot_ensemble_results(l1All_results, save_path=plots_path, seed_scheme='l1All')

    print(f'evals phase 1 complete.')
    print(f'evals phase 2 of 2 initiated.')



    with open(f'{plots_path}/l1Seeded_stability_results.json', 'w') as f:
        json.dump(l1Seeded_results, f)

    with open(f'{plots_path}/l1All_stability_results.json', 'w') as f:
        json.dump(l1All_results, f)
    
    print(f'evals phase 2 complete, powering down...')
    print(f'goodbye.')
    return datasets, l1Seeded_results, l1All_results
# -----------------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------------
def save_subdicts_to_h5(main_dict, save_dir):
    """
    Saves each sub-dictionary of NumPy arrays in the main_dict to separate HDF5 files.
    
    Args:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
        save_dir (str): The directory where the HDF5 files will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for sub_dict_name, sub_dict in main_dict.items():
        file_path = os.path.join(save_dir, f"{sub_dict_name}.h5")
        with h5py.File(file_path, 'w') as f:
            for key, arr in sub_dict.items():
                f.create_dataset(key, data=arr)
        print(f"Saved {sub_dict_name} to {file_path}")
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def load_subdicts_from_h5(save_dir):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5"):
            sub_dict_name = os.path.splitext(filename)[0]
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------

def create_wrapper_model(preprocessing_model, mse_model):
    raw_inputs = tf.keras.Input(shape=preprocessing_model.input_shape[1:])
    preprocessed_data = preprocessing_model(raw_inputs)
    ad_scores = mse_model(preprocessed_data)
    wrapper_model = tf.keras.Model(inputs=raw_inputs, outputs=ad_scores)

    return wrapper_model

# -----------------------------------------------------------------------------------------
def convert_to_onnx(training_info, data_info, model_version, save_dir, opset=13, input_dim=48, obj_type='HLT'):
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    # Load the model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=input_dim,
        pt_thresholds=data_info['pt_thresholds'],
        pt_scale_factor=data_info['pt_scale_factor'],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        #large_network=large_network,
        saved_model_path=save_path,
        save_version=model_version,
        obj_type=obj_type
    )

    # Next, we create a wrapper model which combines the preprocessing model with the MSE model
    wrapper_model = create_wrapper_model(HLT_preprocessing_model, HLT_MSE_AE)

    wrapper_model.summary()

    # Convert wrapper model to ONNX
    spec = (tf.TensorSpec(wrapper_model.input_shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(wrapper_model, opset=opset, input_signature=spec)
    
    # Save ONNX model
    onnx_file_path = f"{save_dir}/folded_MSE_AE_{model_version}.onnx"
    onnx.save(onnx_model, onnx_file_path)
    print(f"ONNX HLT_AE model saved to: {onnx_file_path}")

# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def compare_tf_with_onnx(datasets: dict, training_info: dict, data_info, model_version, onnx_path, obj_type='HLT'):
    """
    Compares the outputs of TensorFlow and ONNX models.

    Inputs:
        datasets: dictionary containing data to pass through the models.
        training_info: dictionary with model training details.
        model_version: the version of the model to load.
        onnx_path: path where the ONNX models are stored.

    Returns:
        datasets: updated datasets with both TensorFlow and ONNX results.
    """
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    # Load the model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=48,
        pt_thresholds=data_info['pt_thresholds'],
        pt_scale_factor=data_info['pt_scale_factor'],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        #large_network=large_network,
        saved_model_path=save_path,
        save_version=model_version,
        obj_type=obj_type
    )

    # Pass the data through the TensorFlow model
    skip_tags = ['EB_train', 'EB_val']
    for tag, data_dict in datasets.items():
        if tag in skip_tags: continue

        if tag.startswith('k'):

            data_dict[f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(data_dict[f'{obj_type}_data'], verbose=0)
            data_dict[f'{obj_type}_AD_scores'] = HLT_MSE_AE.predict(data_dict[f'{obj_type}_preprocessed_data'], verbose=0)

    # Load the ONNX models
    HLT_onnx_session = rt.InferenceSession(f"{onnx_path}/folded_MSE_AE_{model_version}.onnx")

    # Run inference using the ONNX models and store results
    for tag, dict in datasets.items():
        if tag in skip_tags: continue

        # Run inference on HLT data
        if tag.startswith('k'):
            onnx_inputs_HLT = {HLT_onnx_session.get_inputs()[0].name: dict[f'{obj_type}_data'].astype(np.float32)}
            dict[f'ONNX_{obj_type}_AD_scores'] = HLT_onnx_session.run(None, onnx_inputs_HLT)[0]

    return datasets
# -----------------------------------------------------------------------------------------
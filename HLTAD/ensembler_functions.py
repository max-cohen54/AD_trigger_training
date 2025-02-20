import os
import numpy as np
import tensorflow as tf
import math
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
    allowed_train_data_schemes = ['topo2A_train', 'L1noalg_HLTall', 'topo2A_train+L1noalg_HLTall', 'B+C_loose']
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

    # third event data
    event3_jets = [
        [100, 0.5, 0.5]
    ]
    event3_electrons = [
        [100, 0.51, 0.49], [100, -0.5, -0.5]
    ]
    event3_muons = []
    event3_photons = [
        [100, -0.51, -0.49]
    ]
    event3_met = [0, 0, 0]

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
    event3_array = process_event(event3_jets, event3_electrons, event3_muons, event3_photons, event3_met)
    # Combine into a single (3, 16, 3) array
    combined_array = np.stack([event1_array, event2_array, event3_array])
    
    datasets['kaito'] = {key: value[0:3] for key, value in datasets['topo2A_train'].items()}
    datasets['kaito']['HLT_data'] = combined_array
    datasets['kaito']['L1_data'] = combined_array


    # Now put in the synthetic events to check overlap removal:

    event1 = np.array([
        # Jets (0-5)
        [50, 0.31, 0.0],    # Jet 0 (overlap with electron 0 → should be removed by Rule 4)
        [60, 1.0, 1.0],     # Jet 1 (remains)
        [55, -0.5, 2.0],    # Jet 2 (remains)
        [52, 0.0, -1.0],    # Jet 3 (remains)
        [53, 0.8, 0.8],     # Jet 4 (remains)
        [54, -0.8, -0.8],   # Jet 5 (remains)
        # Electrons (6-8)
        [10, 0.5, 0.0],     # Electron 0 (overlap trigger for muon and photon)
        [12, 2.0, 1.0],     # Electron 1 (remains)
        [11, -1.0, 0.5],    # Electron 2 (remains)
        # Muons (9-11)
        [8, 0.55, 0.05],    # Muon 0 (close to Electron 0, dR ~0.07 → should be removed by Rule 1)
        [9, 3.0, 2.0],      # Muon 1 (remains)
        [10, -1.5, 0.4],    # Muon 2 (remains)
        # Photons (12-14)
        [15, 0.52, -0.05],  # Photon 0 (close to Electron 0, dR ~0.05 → removed by Rule 2)
        [16, 3.05, 2.05],   # Photon 1 (close to Muon 1, dR ~0.07 → removed by Rule 3)
        [14, -2.0, 0.0],    # Photon 2 (remains)
        # MET (15)
        [30, 0.0, 0.0]      # MET (always remains)
    ], dtype=np.float32)

    # Event 2: No overlaps.
    event2 = np.array([
        # Jets (0-5)
        [50, -1.0, -1.0],
        [55, -1.5, -1.5],
        [60, -2.0, -2.0],
        [65, -2.5, -2.5],
        [70, -3.0, -3.0],
        [75, -3.5, -3.5],
        # Electrons (6-8)
        [10, 1.0, 1.0],
        [11, 1.5, 1.5],
        [12, 2.0, 2.0],
        # Muons (9-11)
        [8, 3.0, 3.0],
        [9, 3.5, 3.5],
        [10, 4.0, 4.0],
        # Photons (12-14)
        [15, 5.0, 5.0],
        [16, 5.5, 5.5],
        [17, 6.0, 6.0],
        # MET (15)
        [30, 0.0, 0.0]
    ], dtype=np.float32)

    event3 = np.array([
        # Jets (indices 0–5)
        [50, 0.0, 0.0],    # Jet 0: remains; reference for rule 5 & rule 8.
        [55, 1.0, 1.0],    # Jet 1: will be removed by rule 6 (muon very close).
        [60, -0.5, 2.0],   # Jet 2: used for rule 7.
        [52, 0.0, -1.0],   # Jet 3: remains.
        [53, 0.8, 0.8],    # Jet 4: remains.
        [54, -0.8, -0.8],  # Jet 5: remains.
        # Electrons (indices 6–8)
        [10, 0.3, 0.0],    # Electron 0: near Jet 0 (dR ~0.3) → removed by rule 5.
        [12, 2.0, 1.0],    # Electron 1: remains.
        [11, -1.0, 0.5],   # Electron 2: remains.
        # Muons (indices 9–11)
        [8, 1.1, 1.0],     # Muon 0: very close to Jet 1 → triggers removal of Jet 1 by rule 6.
        [9, 3.0, 2.0],     # Muon 1: remains.
        [10, -0.2, 2.0],   # Muon 2: near Jet 2 (dR ~0.3) → removed by rule 7.
        # Photons (indices 12–14)
        [15, 0.2, 0.0],    # Photon 0: near Jet 0 (dR ~0.2) → removed by rule 8.
        [16, 5.5, 5.5],    # Photon 1: remains.
        [14, -2.0, 0.0],   # Photon 2: remains.
        # MET (index 15)
        [30, 0.0, 0.0]     # MET: remains unchanged.
    ], dtype=np.float32)

    synthetic_events = np.stack([event1, event2, event3], axis=0)  # shape: (3, 16, 3)
    # datasets['synthetic_events'] = {key: value for key, value in datasets['topo2A_train'].items()}
    # datasets['synthetic_events']['HLT_data'] = synthetic_events
    # datasets['synthetic_events']['L1_data'] = synthetic_events

    #remove mc23e for some preliminary testing (to remove later):
    tags_to_remove = []
    for tag in datasets.keys():
        if ('qqa' in tag) or (tag=='jjJZ2') or (tag=='jjJZ1') or (tag=='jjJZ4') or ('Zprime' in tag) or ('ZZ' in tag) or ('A14' in tag) or ('HHbbtt' in tag) or ('HAHM' in tag):
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

    # now combine the other EB runs into EB_test
    tags_to_combine = [key for key in datasets.keys() if "EB" in key and key != 'EB_train']
    datasets = combine_data(datasets, tags_to_combine=tags_to_combine, new_tag='EB_test')

    if train_data_scheme == 'B+C_loose':

        # find the threshold for the loose B+C region
        L1AD_threshold, L1AD_pure_rate , L1AD_total_rate = find_threshold(
            scores=datasets['EB_test']['topo2A_AD_scores'],
            weights=datasets['EB_test']['weights'],
            pass_current_trigs=datasets['EB_test']['passL1'],
            target_rate=13501, # looser threshold
            incoming_rate=31575960
        )

        print(f'B+C Loose: pure rate = {L1AD_pure_rate}, total rate = {L1AD_total_rate}, threshold = {L1AD_threshold}')

        # from the training data, select the events that pass the loose B+C region
        B_C_loose_mask = datasets['topo2A_train']['topo2A_AD_scores'] > L1AD_threshold
        datasets['EB_train'] = {key:value[B_C_loose_mask] for key, value in datasets['topo2A_train'].items()}
        print(f'B+C Loose number of events = {len(datasets["EB_train"]["HLT_data"])}')
        del datasets['topo2A_train']

    
    
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
class OverlapRemovalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def process_event(event):
            # Split the event into objects based on the new ordering.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3)

            # Helper: When an object is "removed" we zero its features.
            # For overlap checks, removed objects should not affect dR calculations.
            # The helper function below replaces removed objects (pt == 0)
            # with sentinel values (eta,phi = 1e6) so that distances computed with them are huge.
            def mask_removed(objs):
                # objs: shape (n, 3) with col0: pt, col1: eta, col2: phi.
                active = tf.expand_dims(objs[:, 0] > 0, axis=-1)  # (n,1)
                sentinel = tf.constant([0.0, 1e6, 1e6], dtype=objs.dtype)
                sentinel = tf.broadcast_to(sentinel, tf.shape(objs))
                return tf.where(active, objs, sentinel)

            # Helper: dphi (accounts for periodicity)
            def dphi(phi1, phi2):
                diff = phi1 - phi2
                return tf.math.floormod(diff + math.pi, 2 * math.pi) - math.pi

            # Helper: pairwise dR calculation between two sets of objects.
            def pairwise_dR(objs1, objs2):
                objs1 = mask_removed(objs1)
                objs2 = mask_removed(objs2)
                eta1 = objs1[:, 1]  # shape (N,)
                phi1 = objs1[:, 2]
                eta2 = objs2[:, 1]  # shape (M,)
                phi2 = objs2[:, 2]
                deta = tf.expand_dims(eta1, axis=1) - tf.expand_dims(eta2, axis=0)  # (N, M)
                dphi_val = dphi(tf.expand_dims(phi1, axis=1), tf.expand_dims(phi2, axis=0))  # (N, M)
                return tf.sqrt(deta**2 + dphi_val**2)  # (N, M)

            # Sequential update functions: each rule zeroes out objects that fail the overlap test.

            # RULE 1: Muon vs Electron: if any electron is within dR < 0.2 of a muon, remove that muon.
            def rule1_update(muons, electrons):
                dr = pairwise_dR(electrons, muons)  # shape (n_elec, n_muon)
                remove = tf.reduce_any(dr < 0.2, axis=0)  # for each muon
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(muons), muons)
            muons = rule1_update(muons, electrons)

            # RULE 2: Photon vs Electron: if any electron is within dR < 0.4 of a photon, remove that photon.
            def rule2_update(photons, electrons):
                dr = pairwise_dR(electrons, photons)  # shape (n_elec, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule2_update(photons, electrons)

            # RULE 3: Photon vs Muon: if any muon is within dR < 0.4 of a photon, remove that photon.
            def rule3_update(photons, muons):
                dr = pairwise_dR(muons, photons)  # shape (n_muon, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule3_update(photons, muons)

            # RULE 4: Jet vs Electron: if any electron is within dR < 0.2 of a jet, remove that jet.
            def rule4_update(jets, electrons):
                dr = pairwise_dR(electrons, jets)  # shape (n_elec, n_jet)
                remove = tf.reduce_any(dr < 0.2, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(jets), jets)
            jets = rule4_update(jets, electrons)

            # RULE 5: Electron vs Jet: if any jet is within dR < 0.4 of an electron, remove that electron.
            def rule5_update(electrons, jets):
                dr = pairwise_dR(jets, electrons)  # shape (n_jet, n_elec)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(electrons), electrons)
            electrons = rule5_update(electrons, jets)

            # RULE 6: Jet vs Muon: if any muon is within dR < 0.2 of a jet, remove that jet.
            def rule6_update(jets, muons):
                dr = pairwise_dR(muons, jets)  # shape (n_muon, n_jet)
                remove = tf.reduce_any(dr < 0.2, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(jets), jets)
            jets = rule6_update(jets, muons)

            # RULE 7: Muon vs Jet: if any jet is within dR < 0.4 of a muon, remove that muon.
            def rule7_update(muons, jets):
                dr = pairwise_dR(jets, muons)  # shape (n_jet, n_muon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(muons), muons)
            muons = rule7_update(muons, jets)

            # RULE 8: Photon vs Jet: if any jet is within dR < 0.4 of a photon, remove that photon.
            def rule8_update(photons, jets):
                dr = pairwise_dR(jets, photons)  # shape (n_jet, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule8_update(photons, jets)

            # Reassemble the event in the new order: jets, electrons, muons, photons, MET.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        return config



class DuplicateRemovalLayer(tf.keras.layers.Layer):
    def __init__(self, duplicate_threshold=0.05, **kwargs):
        """
        Args:
          duplicate_threshold (float): dR threshold below which the later object is considered a duplicate.
        """
        super().__init__(**kwargs)
        self.duplicate_threshold = duplicate_threshold

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def process_event(event):
            # Split the event into objects.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3)

            # Helper: For duplicate removal, we want to ignore objects that have already been removed.
            # If an object is removed (pt==0), we replace its (eta,phi) with sentinel values so that it does not spuriously match.
            def mask_removed(objs):
                # objs: shape (n, 3) with columns: [pt, eta, phi]
                active = tf.expand_dims(objs[:, 0] > 0, axis=-1)  # shape (n,1)
                sentinel = tf.constant([0.0, 1e6, 1e6], dtype=objs.dtype)
                sentinel = tf.broadcast_to(sentinel, tf.shape(objs))
                return tf.where(active, objs, sentinel)

            # Helper: Compute pairwise dR for objects of the same type.
            def pairwise_dR_same(objs):
                # First mask out removed objects.
                objs_masked = mask_removed(objs)
                eta = objs_masked[:, 1]  # shape (n,)
                phi = objs_masked[:, 2]  # shape (n,)
                deta = tf.expand_dims(eta, axis=1) - tf.expand_dims(eta, axis=0)  # (n, n)
                dphi_val = tf.math.floormod(tf.expand_dims(phi, axis=1) - tf.expand_dims(phi, axis=0) + math.pi, 
                                            2 * math.pi) - math.pi  # (n, n)
                return tf.sqrt(deta**2 + dphi_val**2)  # (n, n)

            # Duplicate removal: For objects of the same type, if two objects are within duplicate_threshold,
            # we remove (zero out) the one with the higher index.
            def remove_duplicates(objs, threshold):
                # objs: shape (n, 3)
                n = tf.shape(objs)[0]
                # If there are no objects, just return.
                # (tf.cond is not strictly necessary here since n is small, but we can check if desired.)
                # Compute pairwise distances.
                dr = pairwise_dR_same(objs)  # shape (n, n)
                # Create a lower-triangular mask (excluding the diagonal) so that for each object j, we consider only objects i < j.
                ones = tf.ones_like(dr, dtype=tf.bool)
                # tf.linalg.band_part(ones, 0, -1) gives the upper-triangular part (including diagonal).
                # Its logical_not gives the strictly lower-triangular part.
                lower_tri_exclusive = tf.logical_not(tf.linalg.band_part(ones, 0, -1))
                # For positions where the mask is False, assign a large value so they don't affect our check.
                replaced_dr = tf.where(lower_tri_exclusive, dr, tf.fill(tf.shape(dr), 1e6))
                # For each column j, if any entry in rows 0..j-1 is below the threshold, mark object j as duplicate.
                duplicates = tf.reduce_any(replaced_dr < threshold, axis=0)  # shape (n,)
                # Zero out duplicate objects.
                new_objs = tf.where(tf.expand_dims(duplicates, axis=-1), tf.zeros_like(objs), objs)
                return new_objs

            # Now apply duplicate removal to each object type (except MET).
            jets = remove_duplicates(jets, self.duplicate_threshold)
            electrons = remove_duplicates(electrons, self.duplicate_threshold)
            muons = remove_duplicates(muons, self.duplicate_threshold)
            photons = remove_duplicates(photons, self.duplicate_threshold)

            # Reassemble the event in the original order.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'duplicate_threshold': self.duplicate_threshold})
        return config




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


class ScalePtPerEvent(tf.keras.layers.Layer):
    def __init__(self, target_sum=10.0, epsilon=1e-6, **kwargs):
        """
        Scales the pt's for each event so that the sum of the pt's equals target_sum.
        """
        super().__init__(**kwargs)
        self.target_sum = target_sum
        self.epsilon = epsilon

    def call(self, inputs):

        pts = inputs[:, :, 0]           # Shape: (N, 16)
        other_features = inputs[:, :, 1:]  # Shape: (N, 16, 2)

        # Compute the sum of pts per event (resulting shape: (N, 1))
        sum_pts = tf.reduce_sum(pts, axis=1, keepdims=True)

        # Compute the per-event scaling factor so that new sum equals target_sum.
        scale = self.target_sum / (sum_pts + self.epsilon)  # Shape: (N, 1)

        # Scale the pt's
        scaled_pts = pts * scale  # Still shape: (N, 16)

        # Expand dims to match the other features along the last axis.
        scaled_pts = tf.expand_dims(scaled_pts, axis=-1)  # Shape: (N, 16, 1)

        # Concatenate the scaled pts with the other features
        outputs = tf.concat([scaled_pts, other_features], axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_sum': self.target_sum,
            'epsilon': self.epsilon,
        })
        return config


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
    pt_thresholds, scale_factor, l2_reg=0.01, dropout_rate=0, pt_normalization_type='global_division', overlap_removal=False
):
    # Preprocessing Layers
    # add a initial zero out layer with [30, 15, 15, 15]
    overlap_removal_layer = OverlapRemovalLayer()
    phi_rotation_layer = DeltaPhiPreprocessingLayer()
    met_bias_layer = METBiasMaskLayer()
    zero_out_layer = ZeroOutLowPtLayer(pt_thresholds)
    if pt_normalization_type == 'global_division':
        normalize_pt_layer = NormalizePtLayer(scale_factor)
    elif pt_normalization_type == 'per_event':
        normalize_pt_layer = ScalePtPerEvent(target_sum=10.0)
    flatten_layer = tf.keras.layers.Flatten()

    # Preprocessing Model
    preprocessing_inputs = layers.Input(shape=(num_objects * num_features,))
    unflattened = tf.keras.layers.Reshape((num_objects, num_features))(preprocessing_inputs)
    preprocessed = phi_rotation_layer(unflattened)
    if overlap_removal:
        preprocessed = overlap_removal_layer(preprocessed)
    
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
def load_and_preprocess_and_preprocess(data_info: dict, overlap_removal: bool, obj_type='HLT', tag='EB_test'):

    # Load the data
    datasets, data_info = load_and_preprocess(**data_info)

    
    # Initialize the model
    INPUT_DIM = datasets['EB_test']['HLT_data'].shape[1]
    H_DIM_1 = 100
    H_DIM_2 = 100
    H_DIM_3 = 64
    H_DIM_4 = 32
    LATENT_DIM = 4
        
    HLT_AE, HLT_encoder, HLT_decoder, HLT_MSE_AE, HLT_preprocessing_model = create_large_AE_with_preprocessed_inputs(
        num_objects=16, 
        num_features=3, 
        h_dim_1=H_DIM_1, 
        h_dim_2=H_DIM_2, 
        h_dim_3=H_DIM_3, 
        h_dim_4=H_DIM_4, 
        latent_dim=LATENT_DIM,
        pt_thresholds=data_info['pt_thresholds'],
        scale_factor=data_info['pt_scale_factor'],
        l2_reg=0.01, 
        dropout_rate=0.1,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=overlap_removal
    )

    # Preprocess the data
    datasets[tag][f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(datasets[tag][f'{obj_type}_data'], batch_size=8)

    return datasets
# -----------------------------------------------------------------------------------------
def initialize_model(input_dim, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, saved_model_path=None, save_version=None, obj_type='HLT', pt_normalization_type='global_division', overlap_removal=False):
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
        dropout_rate=dropout_p,
        pt_normalization_type=pt_normalization_type,
        overlap_removal=overlap_removal
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
def train_model(datasets: dict, model_version: str, save_path: str, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, training_weights=True, obj_type='HLT', pt_normalization_type='global_division', overlap_removal=False):
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
        'pt_normalization_type': pt_normalization_type,
        'overlap_removal': overlap_removal
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
def train_multiple_models(datasets: dict, data_info: dict, save_path: str, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, num_trainings=10, training_weights=True, obj_type='HLT', overlap_removal=False):
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
            obj_type=obj_type,
            pt_normalization_type=data_info['pt_normalization_type'],
            overlap_removal=overlap_removal
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
        'obj_type': obj_type,
        'overlap_removal': overlap_removal
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

def raw_efficiencies_plot_from_regions(datasets: dict, save_path: str, save_name: str, seed_scheme: str):
    import numpy as np
    import matplotlib.pyplot as plt
    import ROOT

    if seed_scheme not in ['l1Seeded', 'l1All']:
        raise ValueError(f"Invalid seed_scheme: {seed_scheme}. Must be 'l1Seeded' or 'l1All'.")

    # Dictionaries to store the efficiencies and their uncertainties
    signal_efficiencies = {}
    signal_efficiencies_up = {}
    signal_efficiencies_down = {}

    # Loop over each tag in the datasets dictionary.
    skip_tags = ['EB_train', 'EB_val']
    for tag, data_dict in datasets.items():
        if tag in skip_tags: continue
        region_labels = data_dict['region_labels']
        weights = data_dict['weights']

        # Define the numerator mask: regions containing 'E' or 'F'
        num_mask = np.array([('E' in rl) or ('F' in rl) for rl in region_labels])

        # Define the denominator mask based on seed_scheme.
        if seed_scheme == 'l1Seeded':
            den_mask = np.array([('B' in rl) or ('C' in rl) for rl in region_labels])
        elif seed_scheme == 'l1All':
            den_mask = np.array([('B' in rl) or ('C' in rl) or ('D' in rl) for rl in region_labels])

        # Select the weights for numerator and denominator
        num_weights = weights[num_mask]
        den_weights = weights[den_mask]

        # Skip if the denominator is zero.
        if np.sum(den_weights) == 0:
            continue

        # Create one-bin histograms for numerator and denominator.
        # We use a dummy x-range of [0, 1] and fill at x=0.5.
        h_num = ROOT.TH1F(f"h_num_{tag}", f"Numerator for {tag}", 1, 0, 1)
        h_den = ROOT.TH1F(f"h_den_{tag}", f"Denom for {tag}", 1, 0, 1)

        # Instead of simply adding a single bin content, we fill each event individually.
        # This way the histogram stores the correct sum of weights and also the sum of the squares.
        for w in num_weights:
            h_num.Fill(0.5, w)
        for w in den_weights:
            h_den.Fill(0.5, w)

        # Create a TGraphAsymmErrors from the two histograms using the Bayesian method.
        # The option string sets the confidence level (cl=0.683) and Bayesian prior parameters b(1,1).
        g = ROOT.TGraphAsymmErrors(h_num, h_den, "cl=0.683 b(1,1) mode")

        # Calculate the efficiency from the histograms.
        #efficiency = h_num.GetBinContent(1) / h_den.GetBinContent(1)
        efficiency = np.sum(num_weights) / np.sum(den_weights)
        signal_efficiencies[tag] = efficiency
        signal_efficiencies_up[tag] = g.GetErrorYhigh(0)
        signal_efficiencies_down[tag] = g.GetErrorYlow(0)

    # Create a plot of the efficiencies with asymmetric error bars.
    # For the y-axis, we assign one row per tag.
    tags = list(signal_efficiencies.keys())
    y_positions = np.arange(len(tags))

    plt.figure(figsize=(15, 8))
    # Build the error bar array: first row for lower errors, second row for upper errors.
    yerr = np.array([[signal_efficiencies_down[tag] for tag in tags],
                     [signal_efficiencies_up[tag] for tag in tags]])

    plt.errorbar(list(signal_efficiencies.values()), y_positions, xerr=yerr,
                 fmt='o', color='cornflowerblue', markersize=10, alpha=0.5, capsize=5)

    plt.xlabel('Efficiency', fontsize=15)
    plt.title('Raw Signal Efficiencies', fontsize=16)
    plt.yticks(y_positions, tags)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    # Return the dictionary of efficiencies.
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




def EoverFplusG_plot(datasets: dict, save_path: str, save_name: str):
    
    # Calculate the uncertainty of EoverFplusG:
    # if eff_gain = E / (F + G), and eff = E / (E + F + G), then:
    # eff_gain = eff / (1 - eff)
    # So we'll calculate the uncertainty of eff and then propagate it to eff_gain

    # Dictionaries for the transformed efficiency and its uncertainties.
    EoverFplusG = {}
    uncertainties_up = {}
    uncertainties_down = {}

    # Transformation function: from efficiency (E/(E+F+G)) to E/(F+G)
    def func(x):
        return x / (1 - x)

    # Loop over each tag in the datasets.
    skip_tags = ['EB_train', 'EB_val']
    for tag, data_dict in datasets.items():
        if tag in skip_tags: continue
        region_labels = data_dict['region_labels']
        weights = data_dict['weights']

        # Define the numerator mask: only events with region labels containing 'E'
        num_mask = np.array(['E' in rl for rl in region_labels])
        # Denominator: all events with labels containing 'E', 'F', or 'G'
        den_mask = np.array([('E' in rl) or ('F' in rl) or ('G' in rl) for rl in region_labels])

        # Select the appropriate weights.
        num_weights = weights[num_mask]
        den_weights = weights[den_mask]

        # Skip tags where the denominator sums to zero.
        if np.sum(den_weights) == 0:
            continue

        # Create one-bin histograms for numerator and denominator.
        # We use a dummy x-axis range [0,1] and fill at x=0.5.
        h_num = ROOT.TH1F(f"h_num_{tag}", f"Numerator for {tag}", 1, 0, 1)
        h_den = ROOT.TH1F(f"h_den_{tag}", f"Denom for {tag}", 1, 0, 1)

        # Fill each histogram event-by-event so that the error calculation is correct.
        for w in num_weights:
            h_num.Fill(0.5, w)
        for w in den_weights:
            h_den.Fill(0.5, w)

        # Build a TGraphAsymmErrors using the Bayesian method.
        # The option string "cl=0.683 b(1,1) mode" sets the confidence level and the Bayesian prior.
        g = ROOT.TGraphAsymmErrors(h_num, h_den, "cl=0.683 b(1,1) mode")

        # Compute the efficiency: E/(E+F+G)
        #eff = h_num.GetBinContent(1) / h_den.GetBinContent(1)
        eff = np.sum(num_weights) / np.sum(den_weights)
        # Transform the efficiency to get E/(F+G)
        EoverFplusG[tag] = func(eff)

        # Retrieve the Bayesian uncertainties on the efficiency.
        eff_unc_up = g.GetErrorYhigh(0)
        eff_unc_down = g.GetErrorYlow(0)

        # Propagate the uncertainties through the transformation.
        uncertainties_up[tag] = func(eff + eff_unc_up) - func(eff)
        uncertainties_down[tag] = func(eff) - func(eff - eff_unc_down)

    # Prepare the plotting variables.
    tags = list(EoverFplusG.keys())
    y_positions = np.arange(len(tags))
    # Assemble the asymmetric error array: first row lower, second row upper.
    yerr = np.array([[uncertainties_down[tag] for tag in tags],
                     [uncertainties_up[tag] for tag in tags]])

    # Create the plot.
    plt.figure(figsize=(15, 8))
    plt.errorbar(list(EoverFplusG.values()), y_positions,
                 xerr=yerr,
                 fmt='o', color='cornflowerblue', markersize=10, alpha=0.5, capsize=5)
    plt.xlabel('Efficiency Gain (E/(F+G))', fontsize=15)
    plt.title('Signal Efficiency Gains (E/(F+G))', fontsize=16)
    plt.yticks(y_positions, tags)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return EoverFplusG


    
def EoverB_plot(datasets: dict, save_path: str, save_name: str):
    
    # Calculate the uncertainty of EoverB:
    # if eff_gain = E / B, and eff = E / (E + B), then:
    # eff_gain = eff / (1 - eff)
    # So we'll calculate the uncertainty of eff and then propagate it to eff_gain

    # Dictionaries for the transformed efficiency and its uncertainties.
    EoverB = {}
    uncertainties_up = {}
    uncertainties_down = {}

    # Transformation function: from efficiency (E/(E+B)) to E/B
    def func(x):
        return x / (1 - x)

    # Loop over each tag in the datasets.
    skip_tags = ['EB_train', 'EB_val']
    for tag, data_dict in datasets.items():
        if (tag in skip_tags) or (tag.startswith('phys')): continue
        region_labels = data_dict['region_labels']
        weights = data_dict['weights']

        # Define the numerator mask: only events with region labels containing 'E'
        num_mask = np.array(['E' in rl for rl in region_labels])
        # Denominator: all events with labels containing 'E' or 'B'
        den_mask = np.array([('E' in rl) or ('B' in rl) for rl in region_labels])

        # Select the appropriate weights.
        num_weights = weights[num_mask]
        den_weights = weights[den_mask]

        # Skip tags where the denominator sums to zero.
        if np.sum(den_weights) == 0:
            continue

        # Create one-bin histograms for numerator and denominator.
        # We use a dummy x-axis range [0,1] and fill at x=0.5.
        h_num = ROOT.TH1F(f"h_num_{tag}", f"Numerator for {tag}", 1, 0, 1)
        h_den = ROOT.TH1F(f"h_den_{tag}", f"Denom for {tag}", 1, 0, 1)

        # Fill each histogram event-by-event so that the error calculation is correct.
        for w in num_weights:
            h_num.Fill(0.5, w)
        for w in den_weights:
            h_den.Fill(0.5, w)

        # Build a TGraphAsymmErrors using the Bayesian method.
        # The option string "cl=0.683 b(1,1) mode" sets the confidence level and the Bayesian prior.
        g = ROOT.TGraphAsymmErrors(h_num, h_den, "cl=0.683 b(1,1) mode")

        # Compute the efficiency: E/(E+B)
        #eff = h_num.GetBinContent(1) / h_den.GetBinContent(1)
        eff = np.sum(num_weights) / np.sum(den_weights)
        # Transform the efficiency to get E/B
        EoverB[tag] = func(eff)

        # Retrieve the Bayesian uncertainties on the efficiency.
        eff_unc_up = g.GetErrorYhigh(0)
        eff_unc_down = g.GetErrorYlow(0)

        # Propagate the uncertainties through the transformation.
        uncertainties_up[tag] = func(eff + eff_unc_up) - func(eff)
        uncertainties_down[tag] = func(eff) - func(eff - eff_unc_down)

    # Prepare the plotting variables.
    tags = list(EoverB.keys())
    y_positions = np.arange(len(tags))
    # Assemble the asymmetric error array: first row lower, second row upper.
    yerr = np.array([[uncertainties_down[tag] for tag in tags],
                     [uncertainties_up[tag] for tag in tags]])

    # Create the plot.
    plt.figure(figsize=(15, 8))
    plt.errorbar(list(EoverB.values()), y_positions,
                 xerr=yerr,
                 fmt='o', color='cornflowerblue', markersize=10, alpha=0.5, capsize=5)
    plt.xlabel('E/B', fontsize=15)
    plt.title('E/B', fontsize=16)
    plt.yticks(y_positions, tags)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return EoverB


def E_conditional_overB_plot(datasets: dict, save_path: str, save_name: str):
    
    # Calculate the uncertainty of E conditional over B, where E_conditional=E_B is the number of events in E that came from B (instead of C):
    # if eff_gain = E_B / B, and eff = E_B / (E_B + B), then:
    # eff_gain = eff / (1 - eff)
    # So we'll calculate the uncertainty of eff and then propagate it to eff_gain

    # Dictionaries for the transformed efficiency and its uncertainties.
    E_conditional_overB = {}
    uncertainties_up = {}
    uncertainties_down = {}

    # Transformation function: from efficiency (E_B/(E_B+B)) to E_B/B
    def func(x):
        return x / (1 - x)

    # Loop over each tag in the datasets.
    skip_tags = ['EB_train', 'EB_val']
    for tag, data_dict in datasets.items():
        if (tag in skip_tags) or (tag.startswith('phys')): continue
        region_labels = data_dict['region_labels']
        weights = data_dict['weights']

        # Define the numerator mask: only events with region labels containing 'E' and 'B' (E_B)
        num_mask = np.array([('E' in rl) and ('B' in rl) for rl in region_labels])
        # Denominator: all events with labels containing 'B'
        den_mask = np.array([('B' in rl) for rl in region_labels])

        # Select the appropriate weights.
        num_weights = weights[num_mask]
        den_weights = weights[den_mask]

        # Skip tags where the denominator sums to zero.
        if np.sum(den_weights) == 0:
            continue

        # Create one-bin histograms for numerator and denominator.
        # We use a dummy x-axis range [0,1] and fill at x=0.5.
        h_num = ROOT.TH1F(f"h_num_{tag}", f"Numerator for {tag}", 1, 0, 1)
        h_den = ROOT.TH1F(f"h_den_{tag}", f"Denom for {tag}", 1, 0, 1)

        # Fill each histogram event-by-event so that the error calculation is correct.
        for w in num_weights:
            h_num.Fill(0.5, w)
        for w in den_weights:
            h_den.Fill(0.5, w)

        # Build a TGraphAsymmErrors using the Bayesian method.
        # The option string "cl=0.683 b(1,1) mode" sets the confidence level and the Bayesian prior.
        g = ROOT.TGraphAsymmErrors(h_num, h_den, "cl=0.683 b(1,1) mode")

        # Compute the efficiency: E/(E+B)
        #eff = h_num.GetBinContent(1) / h_den.GetBinContent(1)
        eff = np.sum(num_weights) / np.sum(den_weights)
        # Transform the efficiency to get E/B
        E_conditional_overB[tag] = func(eff)

        # Retrieve the Bayesian uncertainties on the efficiency.
        eff_unc_up = g.GetErrorYhigh(0)
        eff_unc_down = g.GetErrorYlow(0)

        # Propagate the uncertainties through the transformation.
        uncertainties_up[tag] = func(eff + eff_unc_up) - func(eff)
        uncertainties_down[tag] = func(eff) - func(eff - eff_unc_down)

    # Prepare the plotting variables.
    tags = list(E_conditional_overB.keys())
    y_positions = np.arange(len(tags))
    # Assemble the asymmetric error array: first row lower, second row upper.
    yerr = np.array([[uncertainties_down[tag] for tag in tags],
                     [uncertainties_up[tag] for tag in tags]])

    # Create the plot.
    plt.figure(figsize=(15, 8))
    plt.errorbar(list(E_conditional_overB.values()), y_positions,
                 xerr=yerr,
                 fmt='o', color='cornflowerblue', markersize=10, alpha=0.5, capsize=5)
    plt.xlabel('E_B/B', fontsize=15)
    plt.title('E_B/B', fontsize=16)
    plt.yticks(y_positions, tags)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    return E_conditional_overB

    

def efficiency_vs_variable_plot(datasets: dict, save_path: str, save_name: str, obj_type: str, seed_scheme: str):

    # idxs = ['B' in label for label in datasets['EB_test']['region_labels']] | ['C' in label for label in datasets['EB_test']['region_labels']]
    # anomalous = (['E' in label for label in datasets['EB_test']['region_labels']] | ['F' in label for label in datasets['EB_test']['region_labels']])

    pileups = datasets['EB_test']['pileups']
    leading_jet_pt = datasets['EB_test'][f'{obj_type}_data'][:, 0]
    MET_pt = datasets['EB_test'][f'{obj_type}_preprocessed_data'][:, -3]
    jet_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 0:18:3], axis=1)
    el_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 18:27:3], axis=1)
    mu_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 27:36:3], axis=1)
    ph_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 36:45:3], axis=1)
    weights = datasets['EB_test']['weights']

    # Define bins for each variable
    pileup_bins = np.linspace(np.min(pileups[pileups != 0])-5, np.max(pileups)+5, 35)
    jet_pt_bins = np.linspace(np.min(leading_jet_pt)-5, np.percentile(leading_jet_pt, 75)+300, 35)
    MET_pt_bins = np.linspace(np.min(MET_pt)-5, np.percentile(MET_pt, 75)+300, 35)
    multiplicity_bins = np.arange(0, 7, 1, dtype=np.float32)

    # Initialize TEfficiency for pileup
    h_total_pileup = ROOT.TH1F("h_total_pileup", "Total Events Pileup", len(pileup_bins)-1, pileup_bins)
    h_pass_pileup = ROOT.TH1F("h_pass_pileup", "Passed Events Pileup", len(pileup_bins)-1, pileup_bins)

    # Initialize TEfficiency for leading jet pt
    h_total_jet_pt = ROOT.TH1F("h_total_jet_pt", "Total Events Jet Pt", len(jet_pt_bins)-1, jet_pt_bins)
    h_pass_jet_pt = ROOT.TH1F("h_pass_jet_pt", "Passed Events Jet Pt", len(jet_pt_bins)-1, jet_pt_bins)

    # Initialize TEfficiency for MET pt
    h_total_MET_pt = ROOT.TH1F("h_total_MET_pt", "Total Events MET Pt", len(MET_pt_bins)-1, MET_pt_bins)
    h_pass_MET_pt = ROOT.TH1F("h_pass_MET_pt", "Passed Events MET Pt", len(MET_pt_bins)-1, MET_pt_bins)

    # Initialize TEfficiency for object multiplicities
    h_total_jet_multiplicity = ROOT.TH1F("h_total_jet_multiplicity", "Total Events Jet Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)
    h_pass_jet_multiplicity = ROOT.TH1F("h_pass_jet_multiplicity", "Passed Events Jet Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)

    h_total_el_multiplicity = ROOT.TH1F("h_total_el_multiplicity", "Total Events Electron Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)
    h_pass_el_multiplicity = ROOT.TH1F("h_pass_el_multiplicity", "Passed Events Electron Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)

    h_total_mu_multiplicity = ROOT.TH1F("h_total_mu_multiplicity", "Total Events Muon Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)
    h_pass_mu_multiplicity = ROOT.TH1F("h_pass_mu_multiplicity", "Passed Events Muon Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)

    h_total_ph_multiplicity = ROOT.TH1F("h_total_ph_multiplicity", "Total Events Photon Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)
    h_pass_ph_multiplicity = ROOT.TH1F("h_pass_ph_multiplicity", "Passed Events Photon Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)

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
            h_total_jet_multiplicity.Fill(jet_multiplicity[i], datasets['EB_test']['weights'][i])
            h_total_el_multiplicity.Fill(el_multiplicity[i], datasets['EB_test']['weights'][i])
            h_total_mu_multiplicity.Fill(mu_multiplicity[i], datasets['EB_test']['weights'][i])
            h_total_ph_multiplicity.Fill(ph_multiplicity[i], datasets['EB_test']['weights'][i])
        
        # Fill the pass histograms with events passing HLTAD (regions E and F)
        if HLTAD_mask[i]:
            h_pass_pileup.Fill(datasets['EB_test']['pileups'][i], datasets['EB_test']['weights'][i])
            h_pass_jet_pt.Fill(leading_jet_pt[i], datasets['EB_test']['weights'][i])
            h_pass_MET_pt.Fill(MET_pt[i], datasets['EB_test']['weights'][i])
            h_pass_jet_multiplicity.Fill(jet_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_el_multiplicity.Fill(el_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_mu_multiplicity.Fill(mu_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_ph_multiplicity.Fill(ph_multiplicity[i], datasets['EB_test']['weights'][i])

    # Create TEfficiency objects
    # eff_pileup = ROOT.TEfficiency(h_pass_pileup, h_total_pileup)
    # eff_jet_pt = ROOT.TEfficiency(h_pass_jet_pt, h_total_jet_pt)
    # eff_MET_pt = ROOT.TEfficiency(h_pass_MET_pt, h_total_MET_pt)

    eff_pileup = ROOT.TGraphAsymmErrors(h_pass_pileup, h_total_pileup, "cl=0.683 b(1,1) mode")
    eff_jet_pt = ROOT.TGraphAsymmErrors(h_pass_jet_pt, h_total_jet_pt, "cl=0.683 b(1,1) mode")
    eff_MET_pt = ROOT.TGraphAsymmErrors(h_pass_MET_pt, h_total_MET_pt, "cl=0.683 b(1,1) mode")
    eff_jet_multiplicity = ROOT.TGraphAsymmErrors(h_pass_jet_multiplicity, h_total_jet_multiplicity, "cl=0.683 b(1,1) mode")
    eff_el_multiplicity = ROOT.TGraphAsymmErrors(h_pass_el_multiplicity, h_total_el_multiplicity, "cl=0.683 b(1,1) mode")
    eff_mu_multiplicity = ROOT.TGraphAsymmErrors(h_pass_mu_multiplicity, h_total_mu_multiplicity, "cl=0.683 b(1,1) mode")
    eff_ph_multiplicity = ROOT.TGraphAsymmErrors(h_pass_ph_multiplicity, h_total_ph_multiplicity, "cl=0.683 b(1,1) mode")

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
        c_MET_pt.SaveAs(f'{save_path}/{save_name}_MET.png')

        # Plot efficiency vs jet multiplicity using ROOT
        c_jet_multiplicity = ROOT.TCanvas("c_jet_multiplicity", "Efficiency vs Jet Multiplicity", 800, 600)
        eff_jet_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Jet Multiplicity;Jet Multiplicity;Efficiency")
        eff_jet_multiplicity.Draw("AP")
        c_jet_multiplicity.SaveAs(f'{save_path}/{save_name}_jet_multiplicity.png')

        # Plot efficiency vs electron multiplicity using ROOT
        c_el_multiplicity = ROOT.TCanvas("c_el_multiplicity", "Efficiency vs Electron Multiplicity", 800, 600)
        eff_el_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Electron Multiplicity;Electron Multiplicity;Efficiency")
        eff_el_multiplicity.Draw("AP")
        c_el_multiplicity.SaveAs(f'{save_path}/{save_name}_el_multiplicity.png')

        # Plot efficiency vs muon multiplicity using ROOT
        c_mu_multiplicity = ROOT.TCanvas("c_mu_multiplicity", "Efficiency vs Muon Multiplicity", 800, 600)
        eff_mu_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Muon Multiplicity;Muon Multiplicity;Efficiency")
        eff_mu_multiplicity.Draw("AP")
        c_mu_multiplicity.SaveAs(f'{save_path}/{save_name}_mu_multiplicity.png')

        # Plot efficiency vs photon multiplicity using ROOT
        c_ph_multiplicity = ROOT.TCanvas("c_ph_multiplicity", "Efficiency vs Photon Multiplicity", 800, 600)
        eff_ph_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Photon Multiplicity;Photon Multiplicity;Efficiency")
        eff_ph_multiplicity.Draw("AP")
        c_ph_multiplicity.SaveAs(f'{save_path}/{save_name}_ph_multiplicity.png')

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
    signal_efficiencies = raw_efficiencies_plot_from_regions(datasets, save_path=save_path, save_name=f'Efficiencies_region_counts_{model_version}_{seed_scheme}', seed_scheme=seed_scheme)

    # Plot the efficiency gains
    EoverFplusG = EoverFplusG_plot(datasets, save_path=save_path, save_name=f'EoverFplusG_{model_version}_{seed_scheme}')
    efficiency_gain_plot(region_counts, save_path=save_path, save_name=f'Efficiency_gains_{model_version}_{seed_scheme}', target_rate=target_HLTAD_rate)

    # Plot the efficiency vs variable
    efficiency_vs_variable_plot(datasets, save_path=save_path, save_name=f'Efficiency_plot_{model_version}_{seed_scheme}', obj_type=obj_type, seed_scheme=seed_scheme)

    # Plot E over B plots
    EoverB = EoverB_plot(datasets, save_path=save_path, save_name=f'EoverB_{model_version}_{seed_scheme}')
    E_conditional_overB = E_conditional_overB_plot(datasets, save_path=save_path, save_name=f'E_conditional_overB_{model_version}_{seed_scheme}')
    return signal_efficiencies, EoverFplusG, EoverB


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
    plt.ylabel('Efficiency')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

def ensemble_EoverB_plot(EoverBs: dict, save_path: str, save_name: str):
    EoverBs = {tag: [EoverB[tag] for EoverB in EoverBs] for tag in EoverBs[0].keys()}

    plt.figure(figsize=(12, 6))
    plt.boxplot(EoverBs.values(), labels=EoverBs.keys())
    plt.title(f'Distribution of E over Bs')
    plt.ylabel('E/B')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

def plot_ensemble_results(results: dict, save_path: str, seed_scheme: str):
    efficiency_gains = results['efficiency_gains']
    efficiencies = results['efficiencies']
    EoverBs = results['EoverBs']
    ensemble_efficiency_gain_plot(efficiency_gains, save_path, save_name=f'Efficiency_gains_distribution_{seed_scheme}')
    ensemble_raw_efficiencies_plot(efficiencies, save_path, save_name=f'Efficiencies_distribution_{seed_scheme}')
    ensemble_EoverB_plot(EoverBs, save_path, save_name=f'EoverB_distribution_{seed_scheme}')



    
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
        'efficiencies': [],
        'EoverBs': []
    }

    l1All_results = {
        'region_counts': [],
        'efficiency_gains': [],
        'efficiencies': [],
        'EoverBs': []
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
            obj_type=obj_type,
            pt_normalization_type=data_info['pt_normalization_type'],
            overlap_removal=training_info['overlap_removal']
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
        signal_efficiencies, EoverFplusG, EoverB = plot_individual_model_results(
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
        l1Seeded_results['EoverBs'].append(EoverB)
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
        signal_efficiencies, EoverFplusG, EoverB = plot_individual_model_results(
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
        l1All_results['EoverBs'].append(EoverB)
    
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

def load_and_inference(training_info: dict, data_info: dict, target_rate: int=10, L1AD_rate: int=1000, obj_type='HLT', save_version:int=0, tag='all', seed_scheme:str='l1Seeded'):

    if seed_scheme != 'l1Seeded':
        raise ValueError(f'other seed schemes not yet implemented')
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    #large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    datasets, data_info = load_and_preprocess(**data_info)


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
        save_version=save_version,
        obj_type=obj_type,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal']
    )

    # Pass the data through the model
    skip_tags = ['EB_train', 'EB_val']
    if tag == 'all':
        good_tags = [tag for tag in datasets.keys() if tag not in skip_tags]
    else:
        good_tags = [tag, 'EB_test']

    for tag in good_tags:
        data_dict = datasets[tag]

        # Preprocess the data
        data_dict[f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(data_dict[f'{obj_type}_data'], verbose=0, batch_size=8)

        # Calculate the AD scores
        data_dict[f'{obj_type}_AD_scores'] = HLT_MSE_AE.predict(data_dict[f'{obj_type}_preprocessed_data'], batch_size=8, verbose=0)

    
    # Calculate the L1AD threshold and rates
    L1AD_threshold, L1AD_pure_rate , L1AD_total_rate = find_threshold(
        scores=datasets['EB_test']['topo2A_AD_scores'],
        weights=datasets['EB_test']['weights'],
        pass_current_trigs=datasets['EB_test']['passL1'],
        target_rate=L1AD_rate,
        incoming_rate=31575960
    )


    if seed_scheme == 'l1Seeded':
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
        for tag in good_tags:
            data_dict = datasets[tag]
            
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
        
    return datasets, region_counts




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
        obj_type=obj_type,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal']
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
        obj_type=obj_type,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal']
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
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
from matplotlib.colors import LogNorm

distinct_colors = [
        '#e41a1c',  # red
        '#377eb8',  # blue
        '#4daf4a',  # green
        '#984ea3',  # purple
        '#ff7f00',  # orange
        '#a65628',  # brown
        '#f781bf',  # pink
        '#999999',  # grey
        '#dede00',  # yellow
        '#458b74',  # sea green
    ]

# -----------------------------------------------------------------------------------------
def load_subdicts_from_h5(save_dir, tags_to_use=None):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5") and not filename.startswith('.'):

            # Make sure it’s a file (not directory!)
            if not os.path.isfile(file_path):
                continue

            
            sub_dict_name = os.path.splitext(filename)[0]
            if tags_to_use is not None and sub_dict_name not in tags_to_use:
                continue
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def combine_data(datasets, tags_to_combine, new_tag, delete_old_tags=True):
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
    if delete_old_tags:
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
def kinematic_plot(datasets: dict, obj_idx: int, plots_path: str, save_name: str, tags_to_use: list):
    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Create a color map for consistent colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(tags_to_use))]
    
    # Plot data for each tag with consistent colors
    for tag, color in zip(tags_to_use, colors):
        pts = datasets[tag]['HLT_data'][:, obj_idx, 0]
        etas = datasets[tag]['HLT_data'][:, obj_idx, 1] 
        phis = datasets[tag]['HLT_data'][:, obj_idx, 2]
        weights = datasets[tag]['weights']

        # Use the same color for each tag across all subplots
        ax1.hist(pts, bins=np.linspace(0, 1000, 50), density=True, histtype='step', 
                 linewidth=2.5, fill=False, color=color, label=tag, weights=weights)
        ax2.hist(etas, bins=np.linspace(-5, 5, 50), density=True, histtype='step', 
                 linewidth=2.5, fill=False, color=color, weights=weights)
        ax3.hist(phis, bins=np.linspace(-np.pi, np.pi, 50), density=True, histtype='step', 
                 linewidth=2.5, fill=False, color=color, weights=weights)

    # Set labels and scales
    ax1.set_xlabel(r'$p_T$ [GeV]', fontsize=14)
    ax1.set_ylabel(r'Density', fontsize=14)
    ax1.set_yscale('log')

    ax2.set_xlabel(r'$\eta$', fontsize=14)
    ax2.set_ylabel(r'Density', fontsize=14) 
    ax2.set_yscale('log')

    ax3.set_xlabel(r'$\phi$', fontsize=14)
    ax3.set_ylabel(r'Density', fontsize=14)
    ax3.set_yscale('log')

    # Create a single legend using only the first subplot's handles and labels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14, loc='center', bbox_to_anchor=(0.5, 0.2), 
              ncol=1)
    
    plt.tight_layout()
    # Adjust bottom margin to prevent legend overlap
    plt.subplots_adjust(bottom=0.5)
    
    plt.savefig(os.path.join(plots_path, f'{save_name}.png'))
    plt.close()

def multiplicity_plot(datasets: dict, plots_path: str, save_name: str, tags_to_use: list):

    plt.figure(figsize=(15, 10))

    for tag in tags_to_use:
        multiplicity = np.count_nonzero(datasets[tag]['HLT_data'], axis=(1,2)) // 3 # number of features per object
        weights = datasets[tag]['weights']
        plt.hist(multiplicity, bins=np.arange(0, 17, 1), density=True, histtype='step', linewidth=2.5, fill=False, label=tag, weights=weights)

    plt.xlabel(r'Multiplicity', fontsize=14)
    plt.ylabel(r'Density', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(plots_path, f'{save_name}.png'))
    plt.close()

def leading_jet_dR_plot(datasets: dict, plots_path: str, save_name: str, tags_to_use: list):

    plt.figure(figsize=(15, 8))

    for tag in tags_to_use:
        # leading jet
        eta1 = datasets[tag]['HLT_data'][:, 0, 1]
        phi1 = datasets[tag]['HLT_data'][:, 0, 2]
        # subleading jet
        eta2 = datasets[tag]['HLT_data'][:, 1, 1]
        phi2 = datasets[tag]['HLT_data'][:, 1, 2]

        # calculate dR
        dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi # accounts for periodicity of phi
        deta = eta1 - eta2
        dR = np.sqrt(dphi**2 + deta**2)
        weights = datasets[tag]['weights']
        plt.hist(dR, bins=np.linspace(0, 5, 50), density=True, histtype='step', linewidth=2.5, fill=False, label=tag, weights=weights)

    plt.xlabel(r'Leading jets dR', fontsize=14)
    plt.ylabel(r'Density', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(plots_path, f'{save_name}.png'))
    plt.close()

def sphericity_plot(datasets: dict, plots_path: str, save_name: str, tags_to_use: list):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    if 'EB_val' not in tags_to_use:
        tags_to_use.append('EB_val')
    if 'EB_train' not in tags_to_use:
        tags_to_use.append('EB_train')
    if 'EB_test' not in tags_to_use:
        tags_to_use.append('EB_test')

    # Create a color map for consistent colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(tags_to_use))]

    for tag, color in zip(tags_to_use, colors):
        sphericity = []
        aplanarity = []
        planarity = []
        weights = datasets[tag]['weights']
        for i_event in range(len(datasets[tag]['HLT_data'])):
            pts = datasets[tag]['HLT_data'][i_event, :, 0]
            etas = datasets[tag]['HLT_data'][i_event, :, 1]
            phis = datasets[tag]['HLT_data'][i_event, :, 2]

            # Construct momenta
            pxs = pts * np.cos(phis)
            pys = pts * np.sin(phis)
            pzs = pts * np.sinh(etas)
            p2 = pxs**2 + pys**2 + pzs**2

            # build the normalized tensor M
            P2sum = np.sum(p2)
            if P2sum == 0:
                sphericity.append(0)
                aplanarity.append(0)
                planarity.append(0)
                continue

            M = np.zeros((3,3))
            for x,y,z,w in zip(pxs,pys,pzs,p2):
                vec = np.array([x,y,z])
                M += np.outer(vec,vec)
            M /= P2sum

            # diagonalize
            lam = np.linalg.eigvalsh(M)   # ascending
            lam1, lam2, lam3 = lam[::-1]  # descending

            # event‐shapes
            sphericity.append(1.5*(lam2 + lam3))
            aplanarity.append(1.5*lam3)
            planarity.append(lam2 - lam3)
        
        datasets[tag]['sphericity'] = np.array(sphericity)
        datasets[tag]['aplanarity'] = np.array(aplanarity)
        datasets[tag]['planarity'] = np.array(planarity)
    
        # Remove EB_train and EB_val tags since we don't want to plot those
        if tag in ['EB_train', 'EB_val']:
            continue

        ax1.hist(sphericity, bins=np.linspace(0, 1, 50), density=True, histtype='step', linewidth=2.5, fill=False, label=tag, weights=weights, color=color)
        ax2.hist(aplanarity, bins=np.linspace(0, 1, 50), density=True, histtype='step', linewidth=2.5, fill=False, label=tag, weights=weights, color=color)
        ax3.hist(planarity, bins=np.linspace(0, 1, 50), density=True, histtype='step', linewidth=2.5, fill=False, label=tag, weights=weights, color=color)

    ax1.set_xlabel(r'Sphericity', fontsize=14)
    ax1.set_ylabel(r'Density', fontsize=14)
    ax1.set_yscale('log')

    ax2.set_xlabel(r'Aplanarity', fontsize=14)
    ax2.set_ylabel(r'Density', fontsize=14)
    ax2.set_yscale('log')

    ax3.set_xlabel(r'Planarity', fontsize=14)
    ax3.set_ylabel(r'Density', fontsize=14)
    ax3.set_yscale('log')

    # Place legend below the subplots
    #fig.legend(fontsize=14, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(tags_to_use))

    # Create a single legend using only the first subplot's handles and labels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14, loc='center', bbox_to_anchor=(0.5, 0.2), 
              ncol=1)
    
    plt.tight_layout()
    # Adjust bottom margin to prevent legend overlap
    plt.subplots_adjust(bottom=0.5)
    
    plt.savefig(os.path.join(plots_path, f'{save_name}.png'))
    plt.close()
    
    
    

def plot_input_data(datasets: dict, plots_path: str, tags_to_use: list):
    
    #tags_to_use = [key for key in datasets.keys() if key != 'EB_train' and key != 'EB_val']

    # Plot leading object distributions
    kinematic_plot(datasets, 0, plots_path, save_name='input_leading_jet', tags_to_use=tags_to_use) # leading jet
    kinematic_plot(datasets, 6, plots_path, save_name='input_leading_electron', tags_to_use=tags_to_use) # leading electron
    kinematic_plot(datasets, 9, plots_path, save_name='input_leading_muon', tags_to_use=tags_to_use) # leading muon
    kinematic_plot(datasets, 12, plots_path, save_name='input_leading_photon', tags_to_use=tags_to_use) # leading photon
    
    multiplicity_plot(datasets, plots_path, save_name='input_multiplicity', tags_to_use=tags_to_use)
    
    leading_jet_dR_plot(datasets, plots_path, save_name='input_leading_jet_dR', tags_to_use=tags_to_use)
    
    sphericity_plot(datasets, plots_path, save_name='input_sphericity', tags_to_use=tags_to_use)




# -----------------------------------------------------------------------------------------
def load_and_preprocess(train_data_scheme: str, pt_normalization_type=None, L1AD_rate=1000, pt_thresholds=[50, 0, 0, 0], pt_scale_factor=0.05, comments=None, plots_path=None):
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
    #tcpufit MET: new samples, fixed L1 menu
    datasets = load_subdicts_from_h5('/eos/home-m/mmcohen/ad_trigger_development/data/loaded_and_matched_ntuples/04-16-2025')

    # tcpufit MET: new samples, fixed L1 menu no TLA
    #datasets = load_subdicts_from_h5('/eos/home-m/mmcohen/ad_trigger_development/data/loaded_and_matched_ntuples/04-29-2025_no_TLA')


    # put in Kaito's fake data:
    # First event data
    event1_jets = [
        [130.671, -0.676, 1.139], [64.401, 2.077, -1.519], [42.421, 1.028, -2.132],
        [26.381, 0.512, -2.698], [25.568, 0.048, 0.444]
    ]
    event1_electrons = [
        [73.236, -0.666, 1.112]
    ]
    event1_muons = []  # No muons
    event1_photons = []
    event1_met = [50.266, 0, -2.704]

    # Second event data
    event2_jets = [
        [21.934, -0.644, -2.807], [21.268, 2.118, 1.713]
    ]
    event2_electrons = []
    event2_muons = []  # No muons
    event2_photons = []
    event2_met = [16.403, 0.0, 0.000]

    # third event data
    event3_jets = [
    
    ]
    event3_electrons = [
        
    ]
    event3_muons = []
    event3_photons = [
        
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

    event4 = np.array([
        # Jets (indices 0–5)
        [50, 0.0, 0.0],    
        [50, 0.0, 0.0],   
        [60, -0.5, 2.0],   
        [52, 0.0, -1.0], 
        [53, 0.8, 0.8], 
        [54, -0.8, -0.8],
        # Electrons (indices 6–8)
        [10, 0.3, 0.0],  
        [10, 0.3, 0.0],  
        [11, -1.0, 0.5],  
        # Muons (indices 9–11)
        [8, 1.1, 1.0],   
        [8, 1.1, 1.0],  
        [10, -0.2, 2.0], 
        # Photons (indices 12–14)
        [15, 0.2, 0.0],  
        [15, 0.2, 0.0],    
        [14, -2.0, 0.0],  
        # MET (index 15)
        [30, 0.0, 0.0]     # MET: remains unchanged.
    ], dtype=np.float32)

    synthetic_events = np.stack([event1, event2, event3, event4], axis=0)  # shape: (4, 16, 3)
    #datasets['synthetic_events'] = {key: value for key, value in datasets['topo2A_train'].items()}
    #datasets['synthetic_events']['HLT_data'] = synthetic_events
    #datasets['synthetic_events']['L1_data'] = synthetic_events

    
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


    # remove any datasets that are not in the list
    #tags_to_use = ['EB_test', 'EB_train', 'EB_val', 'mc23e_new_HtoSUEP_ggH_fullhad_125_3p00_3p00_noFilter', 'mc23e_jjJZ2', 'mc23e_new_HAHM_S2Zd4e_60_25_0p1ns', 'mc23e_ttbar_2lep', 'mc23e_new_qqa_Ph25_mRp150_gASp1_qContentUDSC']
    tags_to_use = [tag for tag in datasets.keys()]
    tags_to_delete = [tag for tag in datasets.keys() if tag not in tags_to_use ]
    for tag in tags_to_delete:
        del datasets[tag]

    
    
    # ------------------- 
    
    # # save raw data
    # for tag, data_dict in datasets.items():
    #     datasets[tag]['raw_HLT_data'] = np.copy(data_dict['HLT_data'])
    #     datasets[tag]['raw_L1_data'] = np.copy(data_dict['L1_data'])
    # # -------------------

    # Split the train data into train + val
    indices = np.arange(len(datasets['EB_train']['HLT_data']))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=0)
    
    datasets['EB_val'] = {key:value[val_indices] for key, value in datasets['EB_train'].items()}
    datasets['EB_train'] = {key:value[train_indices] for key, value in datasets['EB_train'].items()}

    # -------------------

    if plots_path is not None:
        # Make input plots
        #tags_to_use=['EB_test', 'mc23e_new_HtoSUEP_ggH_fullhad_125_3p00_3p00_noFilter', 'mc23e_jjJZ2', 'mc23e_new_HAHM_S2Zd4e_60_25_0p1ns', 'mc23e_ttbar_2lep', 'mc23e_new_qqa_Ph25_mRp150_gASp1_qContentUDSC']
        tags_to_use = [tag for tag in datasets.keys()]
        plot_input_data(datasets, plots_path, tags_to_use=tags_to_use)

    # construct some other discriminating varaiables
    for tag, data_dict in datasets.items():
        average_pt = np.mean(data_dict['HLT_data'][:, :, 0], axis=1)
        multiplicity = np.count_nonzero(data_dict['HLT_data'], axis=(1,2)) // 3
        data_dict['pt_mult'] = average_pt * multiplicity # This is negative someitmes because of the -999 MET values
        data_dict['multiplicity'] = multiplicity
        data_dict['leading_jet_pt'] = data_dict['HLT_data'][:, 0, 0]

    # Flatten ndarrays for use in DNN
    for tag, dict in datasets.items():
        for label, data in dict.items():
            if label.endswith('data'):
                datasets[tag][label] = np.reshape(data, newshape=(-1, 48))

    # -------------------



    data_info = {
        'train_data_scheme': train_data_scheme,
        'pt_normalization_type': pt_normalization_type,
        'L1AD_rate': L1AD_rate,
        'pt_thresholds': pt_thresholds,
        'pt_scale_factor': pt_scale_factor,
        'plots_path': plots_path
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



# class ReorderObjectsLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inputs):
#         """
#         inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
#            0-5:   jets
#            6-8:   electrons
#            9-11:  muons
#            12-14: photons
#            15:    MET (untouched)
#         """
#         def process_event(event):
#             # Split the event into its constituent object collections.
#             jets = event[0:6]       # shape (6, 3)
#             electrons = event[6:9]  # shape (3, 3)
#             muons = event[9:12]     # shape (3, 3)
#             photons = event[12:15]  # shape (3, 3)
#             met = event[15:16]      # shape (1, 3)

#             #Helper function that reorders a collection so that non-zero objects (pt > 0)
#             #come first, in their original order, followed by zeroed objects.
#             def reorder_collection(collection):
#                 # Determine which objects are non-zero (based on pt in column 0).
#                 non_zero_mask = collection[:, 0] > 0
#                 non_zero = tf.boolean_mask(collection, non_zero_mask)
#                 zeros = tf.boolean_mask(collection, tf.logical_not(non_zero_mask))
#                 return tf.concat([non_zero, zeros], axis=0)

#             # def reorder_collection(collection):
#             #     condition = tf.reduce_all(tf.equal(collection, 0))
#             #     return tf.cond(
#             #         condition,
#             #         lambda: collection,  # if true, just return collection
#             #         lambda: _do_reorder(collection)  # otherwise, perform the reordering
#             #     )

#             # def _do_reorder(collection):
#             #     non_zero_mask = collection[:, 0] > 0
#             #     non_zero = tf.boolean_mask(collection, non_zero_mask)
#             #     zeros = tf.boolean_mask(collection, tf.logical_not(non_zero_mask))
#             #     return tf.concat([non_zero, zeros], axis=0)

    

#             # Reorder each collection individually.
#             jets = reorder_collection(jets)
#             electrons = reorder_collection(electrons)
#             muons = reorder_collection(muons)
#             photons = reorder_collection(photons)

#             output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
#             tf.debugging.assert_equal(tf.shape(output_event)[0], 16, message="Per-event shape must be 16 rows")
#             # tf.debugging.assert_equal(tf.shape(output_event)[0], 16)
#             # tf.print("Output event shape:", tf.shape(output_event))
#             #output_event.set_shape([16, 3])

#             # Reassemble the event in the original overall ordering.
#             return output_event

#         # Apply the reordering to each event in the batch.
#         outputs = tf.map_fn(process_event, inputs)
#         return outputs

#     def get_config(self):
#         config = super().get_config()
#         return config



class ReorderObjectsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Expected sizes per object group.
        self.expected = {
            'jets': 6,
            'electrons': 3,
            'muons': 3,
            'photons': 3,
            'met': 1
        }

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def pad_collection(collection, expected_size):
            # collection: tensor of shape (n, features)
            # Select objects with pt > 0.
            non_zero = tf.boolean_mask(collection, collection[:, 0] > 0)
            # Truncate if there are more than expected_size.
            non_zero = non_zero[:expected_size]
            # Determine how many rows we got.
            k = tf.shape(non_zero)[0]
            pad_size = expected_size - k
            # Create padding of zeros.
            zeros = tf.zeros((pad_size, tf.shape(collection)[1]), dtype=collection.dtype)
            padded = tf.concat([non_zero, zeros], axis=0)
            # Force the static shape.
            padded.set_shape([expected_size, collection.shape[1]])
            return padded

        def process_event(event):
            # Slice the event into groups.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3) – assumed fixed
            
            # Process each collection to guarantee fixed size.
            jets = pad_collection(jets, self.expected['jets'])
            electrons = pad_collection(electrons, self.expected['electrons'])
            muons = pad_collection(muons, self.expected['muons'])
            photons = pad_collection(photons, self.expected['photons'])
            # MET is assumed to be already fixed.
            
            # Concatenate the groups.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            # Now enforce that each event is exactly 16 rows.
            output_event.set_shape([16, event.shape[1]])
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        # Optionally enforce the batch shape if known, e.g. (None, 16, 3)
        outputs.set_shape([None, 16, inputs.shape[-1]])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'expected': self.expected})
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

        MET_values = tf.where(tf.expand_dims(MET_zeros, axis=-1), tf.constant([[0, 0, 0]], dtype=data.dtype), MET_values)
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
    pt_thresholds, scale_factor, l2_reg=0.01, dropout_rate=0, pt_normalization_type='global_division', overlap_removal=False, duplicate_removal=False
):
    # Preprocessing Layers
    # add a initial zero out layer with [30, 15, 15, 15]
    overlap_removal_layer = OverlapRemovalLayer()
    duplicate_removal_layer = DuplicateRemovalLayer()
    reorder_objects_layer = ReorderObjectsLayer()
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

    if duplicate_removal:
        preprocessed = duplicate_removal_layer(preprocessed)

    if overlap_removal:
        preprocessed = overlap_removal_layer(preprocessed)

    if duplicate_removal or overlap_removal:
        preprocessed = reorder_objects_layer(preprocessed)
    
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
def load_and_preprocess_and_preprocess(data_info: dict, overlap_removal: bool, duplicate_removal: bool, obj_type='HLT', tag='EB_test'):

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
        overlap_removal=overlap_removal,
        duplicate_removal=duplicate_removal
    )

    # Preprocess the data
    datasets[tag][f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(datasets[tag][f'{obj_type}_data'], batch_size=8)

    return datasets
# -----------------------------------------------------------------------------------------
def initialize_model(input_dim, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, saved_model_path=None, save_version=None, obj_type='HLT', pt_normalization_type='global_division', overlap_removal=False, duplicate_removal=False):
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
        overlap_removal=overlap_removal,
        duplicate_removal=duplicate_removal
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
def train_model(datasets: dict, model_version: str, save_path: str, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, training_weights=True, obj_type='HLT', pt_normalization_type='global_division', overlap_removal=False, duplicate_removal=False):
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
        'overlap_removal': overlap_removal,
        'duplicate_removal': duplicate_removal
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
def train_multiple_models(datasets: dict, data_info: dict, save_path: str, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, num_trainings=10, training_weights=True, obj_type='HLT', overlap_removal=False, duplicate_removal=False):
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
            overlap_removal=overlap_removal,
            duplicate_removal=duplicate_removal
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
        'overlap_removal': overlap_removal,
        'duplicate_removal': duplicate_removal
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
    skip_tags = ['EB_train', 'EB_val', bkg_tag, 'kaito', 'HLT_noalg_eb_L1All']
    for tag in datasets.keys():
        if 'ZB' in tag:
            skip_tags.append(tag)
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
    
    # Fix the legend positioning to prevent it from being cut off
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Add tight_layout with padding or adjust figure size to accommodate legend
    #plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for the legend on the right
    
    plt.savefig(f'{save_path}/{save_name}.png', bbox_inches='tight')
    plt.close()

    return signal_efficiencies


def ROC_curve_other_vars_plot(datasets: dict, save_path: str, save_name: str, HLTAD_threshold, bkg_tag='EB_test', obj_type='HLT'):
    
    # Check if the background tag is valid
    if bkg_tag not in datasets.keys():
        raise ValueError(f"Invalid bkg_tag: {bkg_tag}. Must be in the keys of the datasets dictionary.")

    for var_name in ['pt_mult', 'leading_jet_pt', 'sphericity', 'planarity', 'aplanarity', 'multiplicity']:
    
        # Start the plot
        plt.figure(figsize=(15, 8))
        plt.rcParams['axes.linewidth'] = 2.4

        # Get the background scores and weights
        bkg_scores = datasets[bkg_tag][var_name]
        bkg_weights = datasets[bkg_tag]['weights']
        
        # Loop over each tag
        skip_tags = ['EB_train', 'EB_val', bkg_tag, 'kaito', 'HLT_noalg_eb_L1All']
        for tag in datasets.keys():
            if 'ZB' in tag:
                skip_tags.append(tag)
        for tag, data_dict in datasets.items():
            if tag in skip_tags: continue

            # Get the signal scores and weights
            signal_scores = data_dict[var_name]
            signal_weights = data_dict['weights']

            # Combine the background and signal
            combined_scores = np.concatenate((bkg_scores, signal_scores), axis=0)
            combined_weights = np.concatenate((bkg_weights, signal_weights), axis=0)
            combined_labels = np.concatenate((np.zeros_like(bkg_scores), np.ones_like(signal_scores)), axis=0)

            # Use sklearn to calculate the ROC curve
            FPRs, TPRs, thresholds = roc_curve(y_true=combined_labels, y_score=combined_scores, sample_weight=combined_weights)
            AUC = auc(FPRs, TPRs)

            # Add the ROC curve from this tag to the plot
            plt.plot(FPRs, TPRs, label=f'{tag}, AUC={AUC:.3f}', linewidth=1.5)

        # Plot diagonal line
        xx = np.linspace(0, 1, 100)
        plt.plot(xx, xx, color='grey', linestyle='dashed')


        # Aesthetics
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'ROC curves for {var_name}', fontsize=14)
        
        # Fix the legend positioning to prevent it from being cut off
        plt.legend(fontsize=15, bbox_to_anchor=(1.05, 0.5), loc='center left')
        
        # Add tight_layout with padding or adjust figure size to accommodate legend
        #plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Make room for the legend on the right
        
        plt.savefig(f'{save_path}/{save_name}_{var_name}.png', bbox_inches='tight')
        plt.close()

def calculate_efficiencies_other_vars(datasets: dict, results_dict: dict, L1AD_rate=1000):

    for var_name in ['pt_mult', 'leading_jet_pt', 'sphericity', 'planarity', 'aplanarity', 'multiplicity']:

        # Calculate the L1AD threshold and rates
        L1AD_threshold, L1AD_pure_rate , L1AD_total_rate = find_threshold(
            scores=datasets['EB_test']['topo2A_AD_scores'],
            weights=datasets['EB_test']['weights'],
            pass_current_trigs=datasets['EB_test']['passL1'],
            target_rate=L1AD_rate,
            incoming_rate=31575960
        )


        # L1Seeded ---------------------------------------------------------
        pass_L1AD_mask = datasets['EB_test']['topo2A_AD_scores'] >= L1AD_threshold


        var_threshold, var_pure_rate, var_total_rate = find_threshold(
            scores=datasets['EB_test'][var_name][pass_L1AD_mask],
            weights=datasets['EB_test']['weights'][pass_L1AD_mask],
            pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
            target_rate=10,
            incoming_rate=L1AD_total_rate
        )


        for tag in datasets.keys():
            data_dict = datasets[tag]
            
            # will hold the regions that each event falls into. Each event will be in region A
            region_labels = ['A'] * len(data_dict[var_name])

            pass_var = data_dict[var_name] >= var_threshold
            passHLT = data_dict['passHLT']
            passL1AD = data_dict['topo2A_AD_scores'] >= L1AD_threshold
            passL1 = data_dict['passL1']

            # Add letters to strings where conditions are met
            for j, (l1ad, l1, var, hlt) in enumerate(zip(passL1AD, passL1, pass_var, passHLT)):
                if l1ad and not l1:
                    region_labels[j] += 'B'
                if l1ad and l1:
                    region_labels[j] += 'C'
                if not l1ad and l1:
                    region_labels[j] += 'D'
                if l1ad and var and not hlt:
                    region_labels[j] += 'E'
                if l1ad and var and hlt:
                    region_labels[j] += 'F'
                if (l1 or l1ad) and not var and hlt:
                    region_labels[j] += 'G'

        
            EplusF_mask = np.array([('E' in rl) or ('F' in rl) for rl in region_labels])
            BplusC_mask = np.array([('B' in rl) or ('C' in rl) for rl in region_labels])
            FplusG_mask = np.array([('F' in rl) or ('G' in rl) for rl in region_labels])
            E_mask = np.array([('E' in rl) for rl in region_labels])
            E_conditional_mask = np.array([('E' in rl) and ('B' in rl) for rl in region_labels])
            B_mask = np.array([('B' in rl) for rl in region_labels])

            results_dict[tag][f'{var_name}_raw_efficiency'] = np.sum(data_dict['weights'][EplusF_mask]) / np.sum(data_dict['weights'][BplusC_mask]) # E+F / B+C
            results_dict[tag][f'{var_name}_E_conditional_over_B'] = np.sum(data_dict['weights'][E_conditional_mask]) / np.sum(data_dict['weights'][B_mask]) # E_B / B
            results_dict[tag][f'{var_name}_efficiency_gain'] = np.sum(data_dict['weights'][E_mask]) / np.sum(data_dict['weights'][FplusG_mask]) # E / F+G

    return results_dict

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

    if seed_scheme not in ['l1Seeded', 'l1All']:
        raise ValueError(f"Invalid seed_scheme: {seed_scheme}. Must be 'l1Seeded' or 'l1All'.")

    # Dictionaries to store the efficiencies and their uncertainties
    signal_efficiencies = {}
    signal_efficiencies_up = {}
    signal_efficiencies_down = {}

    # Loop over each tag in the datasets dictionary.
    skip_tags = ['EB_train', 'EB_val']
    for tag in datasets.keys():
        if 'ZB' in tag:
            skip_tags.append(tag)
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
    plt.subplots_adjust(left=0.4)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    # Return the dictionary of efficiencies.
    return signal_efficiencies


def raw_HLT_efficiencies_plot(datasets: dict, save_path: str, save_name: str):

    # Dictionaries to store the efficiencies and their uncertainties
    signal_efficiencies = {}
    signal_efficiencies_up = {}
    signal_efficiencies_down = {}

    # Loop over each tag in the datasets dictionary.
    skip_tags = ['EB_train', 'EB_val']
    for tag in datasets.keys():
        if 'ZB' in tag:
            skip_tags.append(tag)
    for tag, data_dict in datasets.items():
        if tag in skip_tags: continue
        region_labels = data_dict['region_labels']
        weights = data_dict['weights']

        # Define the numerator mask: regions containing 'F' or 'G'
        num_mask = np.array([('F' in rl) or ('G' in rl) for rl in region_labels])

        # Define the denominator mask
        den_mask = np.array([('C' in rl) or ('D' in rl) for rl in region_labels])
    

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
    plt.title('Raw HLT Efficiency', fontsize=16)
    plt.yticks(y_positions, tags)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.subplots_adjust(left=0.4)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
    return signal_efficiencies

def energy_scale_plot(datasets: dict, save_path: str, save_name: str):
    skip_tags = ['EB_train', 'EB_val']
    tags = [tag for tag in datasets.keys() if tag not in skip_tags]
    energy_scales = np.array([np.mean(datasets[tag]['HLT_data'][:, 0]) for tag in tags]) # average of the leading jet pt
    energy_scales /= np.max(energy_scales) # normalize to max of 1

    plt.figure(figsize=(15, 8))
    plt.scatter(energy_scales, tags, color='cornflowerblue', alpha=0.5)
    plt.xlabel('Energy Scale', fontsize=15)
    plt.title('Energy Scale', fontsize=16)
    plt.subplots_adjust(left=0.4)
    plt.grid()
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()





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
    for tag in datasets.keys():
        if 'ZB' in tag:
            skip_tags.append(tag)
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
    plt.subplots_adjust(left=0.4)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

    print(f'inside EoverFplusG_plot:')
    for key, value in EoverFplusG.items():
        print(f'{key}: {value}')
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
    sphericity = datasets['EB_test']['sphericity']
    leading_jet_pt = datasets['EB_test'][f'{obj_type}_data'][:, 0] * 20
    MET_pt = datasets['EB_test'][f'{obj_type}_data'][:, -3] * 20
    jet_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 0:18:3], axis=1)
    el_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 18:27:3], axis=1)
    mu_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 27:36:3], axis=1)
    ph_multiplicity = np.count_nonzero(datasets['EB_test'][f'{obj_type}_data'][:, 36:45:3], axis=1)
    weights = datasets['EB_test']['weights']
    planarity = datasets['EB_test']['planarity']
    aplanarity = datasets['EB_test']['aplanarity']
    pt_mult = datasets['EB_test']['pt_mult']
    multiplicity = datasets['EB_test']['multiplicity']

    # Define bins for each variable
    pileup_bins = np.linspace(np.min(pileups[pileups != 0])-5, np.max(pileups)+5, 35)
    jet_pt_bins = np.linspace(np.min(leading_jet_pt)-5, np.percentile(leading_jet_pt, 75)+300, 35)
    MET_pt_bins = np.linspace(np.min(MET_pt)-5, np.percentile(MET_pt, 75)+300, 35)
    multiplicity_bins = np.arange(0, 7, 1, dtype=np.float32)
    sphericity_bins = np.linspace(0, 1, 35)
    aplanarity_bins = np.linspace(0, 1, 35)
    planarity_bins = np.linspace(0, 1, 35)
    pt_mult_bins = np.linspace(np.min(pt_mult)-5, np.percentile(pt_mult, 75)+300, 35)
    multiplicity_bins = np.arange(0, 17, 1, dtype=np.float32)
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

    h_total_multiplicity = ROOT.TH1F("h_total_multiplicity", "Total Events Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)
    h_pass_multiplicity = ROOT.TH1F("h_pass_multiplicity", "Passed Events Multiplicity", len(multiplicity_bins)-1, multiplicity_bins)

    # Initialize TEfficiency for sphericity
    h_total_sphericity = ROOT.TH1F("h_total_sphericity", "Total Events Sphericity", len(sphericity_bins)-1, sphericity_bins)
    h_pass_sphericity = ROOT.TH1F("h_pass_sphericity", "Passed Events Sphericity", len(sphericity_bins)-1, sphericity_bins)

    # Initialize TEfficiency for aplanarity
    h_total_aplanarity = ROOT.TH1F("h_total_aplanarity", "Total Events Aplanarity", len(aplanarity_bins)-1, aplanarity_bins)
    h_pass_aplanarity = ROOT.TH1F("h_pass_aplanarity", "Passed Events Aplanarity", len(aplanarity_bins)-1, aplanarity_bins)

    # Initialize TEfficiency for planarity
    h_total_planarity = ROOT.TH1F("h_total_planarity", "Total Events Planarity", len(planarity_bins)-1, planarity_bins)
    h_pass_planarity = ROOT.TH1F("h_pass_planarity", "Passed Events Planarity", len(planarity_bins)-1, planarity_bins)

    # Initialize TEfficiency for pt_mult
    h_total_pt_mult = ROOT.TH1F("h_total_pt_mult", "Total Events Pt Mult", len(pt_mult_bins)-1, pt_mult_bins)
    h_pass_pt_mult = ROOT.TH1F("h_pass_pt_mult", "Passed Events Pt Mult", len(pt_mult_bins)-1, pt_mult_bins)

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
            h_total_sphericity.Fill(sphericity[i], datasets['EB_test']['weights'][i])
            h_total_aplanarity.Fill(aplanarity[i], datasets['EB_test']['weights'][i])
            h_total_planarity.Fill(planarity[i], datasets['EB_test']['weights'][i])
            h_total_pt_mult.Fill(pt_mult[i], datasets['EB_test']['weights'][i])
            h_total_multiplicity.Fill(multiplicity[i], datasets['EB_test']['weights'][i])

        # Fill the pass histograms with events passing HLTAD (regions E and F)
        if HLTAD_mask[i]:
            h_pass_pileup.Fill(datasets['EB_test']['pileups'][i], datasets['EB_test']['weights'][i])
            h_pass_jet_pt.Fill(leading_jet_pt[i], datasets['EB_test']['weights'][i])
            h_pass_MET_pt.Fill(MET_pt[i], datasets['EB_test']['weights'][i])
            h_pass_jet_multiplicity.Fill(jet_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_el_multiplicity.Fill(el_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_mu_multiplicity.Fill(mu_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_ph_multiplicity.Fill(ph_multiplicity[i], datasets['EB_test']['weights'][i])
            h_pass_sphericity.Fill(sphericity[i], datasets['EB_test']['weights'][i])
            h_pass_aplanarity.Fill(aplanarity[i], datasets['EB_test']['weights'][i])
            h_pass_planarity.Fill(planarity[i], datasets['EB_test']['weights'][i])
            h_pass_pt_mult.Fill(pt_mult[i], datasets['EB_test']['weights'][i])
            h_pass_multiplicity.Fill(multiplicity[i], datasets['EB_test']['weights'][i])

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
    eff_sphericity = ROOT.TGraphAsymmErrors(h_pass_sphericity, h_total_sphericity, "cl=0.683 b(1,1) mode")
    eff_aplanarity = ROOT.TGraphAsymmErrors(h_pass_aplanarity, h_total_aplanarity, "cl=0.683 b(1,1) mode")
    eff_planarity = ROOT.TGraphAsymmErrors(h_pass_planarity, h_total_planarity, "cl=0.683 b(1,1) mode")
    eff_pt_mult = ROOT.TGraphAsymmErrors(h_pass_pt_mult, h_total_pt_mult, "cl=0.683 b(1,1) mode")
    eff_multiplicity = ROOT.TGraphAsymmErrors(h_pass_multiplicity, h_total_multiplicity, "cl=0.683 b(1,1) mode")

    # Plot efficiency vs pileup using ROOT
    if save_path is not None:
        c_pileup = ROOT.TCanvas("c_pileup", "Efficiency vs Pileup", 800, 600)
        c_pileup.SetLogy()
        eff_pileup.SetTitle(f"Anomalous Event Efficiency vs Pileup;Pileup;Efficiency")
        eff_pileup.Draw("AP")
        c_pileup.SaveAs(f'{save_path}/{save_name}_pileup.png')

        # Plot efficiency vs leading jet pt using ROOT
        c_jet_pt = ROOT.TCanvas("c_jet_pt", "Efficiency vs Leading Jet Pt", 800, 600)
        c_jet_pt.SetLogy()
        eff_jet_pt.SetTitle(f"Anomalous Event Efficiency vs Leading Jet Pt;Leading Jet Pt;Efficiency")
        eff_jet_pt.Draw("AP")
        c_jet_pt.SaveAs(f'{save_path}/{save_name}_jet_pt.png')

        # Plot efficiency vs MET pt using ROOT
        c_MET_pt = ROOT.TCanvas("c_MET_pt", "Efficiency vs MET Pt", 800, 600)
        c_MET_pt.SetLogy()
        eff_MET_pt.SetTitle(f"Anomalous Event Efficiency vs MET Pt;MET Pt;Efficiency")
        eff_MET_pt.Draw("AP")
        c_MET_pt.SaveAs(f'{save_path}/{save_name}_MET.png')

        # Plot efficiency vs jet multiplicity using ROOT
        c_jet_multiplicity = ROOT.TCanvas("c_jet_multiplicity", "Efficiency vs Jet Multiplicity", 800, 600)
        c_jet_multiplicity.SetLogy()
        eff_jet_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Jet Multiplicity;Jet Multiplicity;Efficiency")
        eff_jet_multiplicity.Draw("AP")
        c_jet_multiplicity.SaveAs(f'{save_path}/{save_name}_jet_multiplicity.png')

        # Plot efficiency vs electron multiplicity using ROOT
        c_el_multiplicity = ROOT.TCanvas("c_el_multiplicity", "Efficiency vs Electron Multiplicity", 800, 600)
        c_el_multiplicity.SetLogy()
        eff_el_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Electron Multiplicity;Electron Multiplicity;Efficiency")
        eff_el_multiplicity.Draw("AP")
        c_el_multiplicity.SaveAs(f'{save_path}/{save_name}_el_multiplicity.png')

        # Plot efficiency vs muon multiplicity using ROOT
        c_mu_multiplicity = ROOT.TCanvas("c_mu_multiplicity", "Efficiency vs Muon Multiplicity", 800, 600)
        c_mu_multiplicity.SetLogy()
        eff_mu_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Muon Multiplicity;Muon Multiplicity;Efficiency")
        eff_mu_multiplicity.Draw("AP")
        c_mu_multiplicity.SaveAs(f'{save_path}/{save_name}_mu_multiplicity.png')

        # Plot efficiency vs photon multiplicity using ROOT
        c_ph_multiplicity = ROOT.TCanvas("c_ph_multiplicity", "Efficiency vs Photon Multiplicity", 800, 600)
        c_ph_multiplicity.SetLogy()
        eff_ph_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Photon Multiplicity;Photon Multiplicity;Efficiency")
        eff_ph_multiplicity.Draw("AP")
        c_ph_multiplicity.SaveAs(f'{save_path}/{save_name}_ph_multiplicity.png')

        # Plot efficiency vs sphericity using ROOT
        c_sphericity = ROOT.TCanvas("c_sphericity", "Efficiency vs Sphericity", 800, 600)
        c_sphericity.SetLogy()
        eff_sphericity.SetTitle(f"Anomalous Event Efficiency vs Sphericity;Sphericity;Efficiency")
        eff_sphericity.Draw("AP")
        c_sphericity.SaveAs(f'{save_path}/{save_name}_sphericity.png')

        # Plot efficiency vs aplanarity using ROOT
        c_aplanarity = ROOT.TCanvas("c_aplanarity", "Efficiency vs Aplanarity", 800, 600)
        c_aplanarity.SetLogy()
        eff_aplanarity.SetTitle(f"Anomalous Event Efficiency vs Aplanarity;Aplanarity;Efficiency")
        eff_aplanarity.Draw("AP")
        c_aplanarity.SaveAs(f'{save_path}/{save_name}_aplanarity.png')

        # Plot efficiency vs planarity using ROOT
        c_planarity = ROOT.TCanvas("c_planarity", "Efficiency vs Planarity", 800, 600)
        c_planarity.SetLogy()
        eff_planarity.SetTitle(f"Anomalous Event Efficiency vs Planarity;Planarity;Efficiency")
        eff_planarity.Draw("AP")
        c_planarity.SaveAs(f'{save_path}/{save_name}_planarity.png')

        # Plot efficiency vs pt_mult using ROOT
        c_pt_mult = ROOT.TCanvas("c_pt_mult", "Efficiency vs Pt Mult", 800, 600)
        c_pt_mult.SetLogy()
        eff_pt_mult.SetTitle(f"Anomalous Event Efficiency vs Pt Mult;Pt Mult;Efficiency")
        eff_pt_mult.Draw("AP")
        c_pt_mult.SaveAs(f'{save_path}/{save_name}_pt_mult.png')

        # Plot efficiency vs multiplicity using ROOT
        c_multiplicity = ROOT.TCanvas("c_multiplicity", "Efficiency vs Multiplicity", 800, 600)
        c_multiplicity.SetLogy()
        eff_multiplicity.SetTitle(f"Anomalous Event Efficiency vs Multiplicity;Multiplicity;Efficiency")
        eff_multiplicity.Draw("AP")
        c_multiplicity.SaveAs(f'{save_path}/{save_name}_multiplicity.png')


def AD_score_2d_hists(datasets: dict, save_path: str, save_name: str):

    

    for tag in datasets.keys():
        if tag == 'EB_train' or tag == 'EB_val':
            continue

        # only use events passing L1AD (regions B and C)
        mask = np.array(['B' in label for label in datasets[tag]['region_labels']]) | np.array(['C' in label for label in datasets[tag]['region_labels']])

        pileups = datasets[tag]['pileups'][mask]
        sphericity = datasets[tag]['sphericity'][mask]
        leading_jet_pt = datasets[tag]['HLT_data'][:, 0][mask] * 20
        MET_pt = datasets[tag]['HLT_data'][:, -3][mask] * 20
        jet_multiplicity = np.count_nonzero(datasets[tag]['HLT_data'][:, 0:18:3][mask], axis=1)
        el_multiplicity = np.count_nonzero(datasets[tag]['HLT_data'][:, 18:27:3][mask], axis=1)
        mu_multiplicity = np.count_nonzero(datasets[tag]['HLT_data'][:, 27:36:3][mask], axis=1)
        ph_multiplicity = np.count_nonzero(datasets[tag]['HLT_data'][:, 36:45:3][mask], axis=1)
        total_multiplicity = np.count_nonzero(datasets[tag]['HLT_data'][:, ::3][mask], axis=1) # every third element is pt
        weights = datasets[tag]['weights'][mask]
        planarity = datasets[tag]['planarity'][mask]
        aplanarity = datasets[tag]['aplanarity'][mask]
        pt_mult = datasets[tag]['pt_mult'][mask]
        scores = datasets[tag]['HLT_AD_scores'][mask]
        multiplicity = datasets[tag]['multiplicity'][mask]
        
        # Define bins for each variable
        sphericity_bins = np.linspace(0, 1, 35)
        aplanarity_bins = np.linspace(0, 1, 35)
        planarity_bins = np.linspace(0, 1, 35)
        if len(pt_mult) == 0:
            continue
        pt_mult_bins = np.linspace(np.min(pt_mult)-5, np.percentile(pt_mult, 75)+300, 35)
        pt_mult_bins = np.linspace(np.min(pt_mult)-5, np.max(pt_mult)+5, 35)
        scores_bins = np.linspace(np.min(scores)-5, np.percentile(scores, 75), 35)
        leading_jet_pt_bins = np.linspace(np.min(leading_jet_pt)-5, np.percentile(leading_jet_pt, 75)+300, 35)
        MET_pt_bins = np.linspace(np.min(MET_pt)-5, np.percentile(MET_pt, 75)+300, 35)
        jet_multiplicity_bins = np.arange(0, 7, 1, dtype=np.float32)
        el_multiplicity_bins = np.arange(0, 4, 1, dtype=np.float32)
        mu_multiplicity_bins = np.arange(0, 4, 1, dtype=np.float32)
        ph_multiplicity_bins = np.arange(0, 4, 1, dtype=np.float32)
        total_multiplicity_bins = np.arange(0, 17, 1, dtype=np.float32)
        pileup_bins = np.arange(40, 70, 1, dtype=np.float32)
        multiplicity_bins = np.arange(0, 17, 1, dtype=np.float32)
        for data, bins, data_name in zip([sphericity, aplanarity, planarity, 
                            pt_mult, leading_jet_pt, MET_pt, 
                            jet_multiplicity, el_multiplicity, mu_multiplicity, 
                            ph_multiplicity, total_multiplicity, pileups, multiplicity], 
                            [sphericity_bins, aplanarity_bins, planarity_bins, 
                            pt_mult_bins, leading_jet_pt_bins, MET_pt_bins, 
                            jet_multiplicity_bins, el_multiplicity_bins, mu_multiplicity_bins, 
                            ph_multiplicity_bins, total_multiplicity_bins, pileup_bins, multiplicity_bins],
                            ['sphericity', 'aplanarity', 'planarity', 
                            'pt_mult', 'leading_jet_pt', 'MET_pt', 
                            'jet_multiplicity', 'el_multiplicity', 'mu_multiplicity', 
                            'ph_multiplicity', 'total_multiplicity', 'pileup', 'multiplicity']):

            plt.figure(figsize=(14, 8))
            plt.rcParams['axes.linewidth'] = 2.4
            
            plt.hist2d(data, scores, bins=[bins, scores_bins], cmap='viridis', norm=LogNorm(), weights=weights)
            plt.colorbar()
            plt.xlabel(f'{data_name}', fontsize=16)
            plt.ylabel(f'AD Score', fontsize=16)
            plt.savefig(f'{save_path}/{tag}_{save_name}_{data_name}.png')
            plt.close()




# -----------------------------------------------------------------------------------------
def get_good_lbs_from_grl(grl_path, run_number):
    """
    Parse the GRL XML file and extract good luminosity block ranges for a specific run.
    
    Args:
        grl_path (str): Path to the GRL XML file
        run_number (int): Run number to extract good LBs for
        
    Returns:
        list: List of tuples (start_lb, end_lb) representing good LB ranges
    """
    import xml.etree.ElementTree as ET
    
    # Parse the XML file
    tree = ET.parse(grl_path)
    root = tree.getroot()
    
    good_lb_ranges = []
    
    # Find the LumiBlockCollection for the specified run
    for lb_collection in root.findall('.//LumiBlockCollection'):
        run = int(lb_collection.find('Run').text)
        
        if run == run_number:
            # Extract all LB ranges for this run
            for lb_range in lb_collection.findall('LBRange'):
                start = int(lb_range.get('Start'))
                end = int(lb_range.get('End'))
                good_lb_ranges.append((start, end))
    
    return good_lb_ranges

def ZB_scores_plot(datasets: dict, save_path: str, save_name: str):

    ZB_datasets = load_subdicts_from_h5('/eos/home-m/mmcohen/ad_trigger_development/data/loaded_and_matched_ntuples/ZB_04-21-2025')

    grl_xml = '/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/GoodRunsLists/data24_13p6TeV/20241118/physics_25ns_data24.xml'

    noiseburst_LBs = {
        474562: [906],
        474679: [463, 464, 465],
        474991: [893, 942, 943],
        475052: [469]
    }

    # Loop over each tag and split into good, bad, and noiseburst LBs
    for tag in ZB_datasets.keys():
        data_dict = ZB_datasets[tag]
        run_num = data_dict['run_numbers'][0]
        good_LBs = get_good_lbs_from_grl(grl_xml, run_num)

        # Create mask for good LBs (in GRL and not in noiseburst list)
        grlGood_mask = np.zeros_like(data_dict['lumiBlocks'], dtype=bool)
        grl_nb_mask = np.zeros_like(data_dict['lumiBlocks'], dtype=bool)
        for i, lb in enumerate(data_dict['lumiBlocks']):

            is_in_grl = any(start <= lb <= end for start, end in good_LBs)
            is_nb = (run_num in noiseburst_LBs and lb in noiseburst_LBs[run_num])
            
            grlGood_mask[i] = is_in_grl
            grl_nb_mask[i] = is_nb

        grlBad_mask = ~grlGood_mask

        ZB_datasets[f'{tag}_grlGood'] = {key: np.array(value)[grlGood_mask] for key, value in data_dict.items()}
        ZB_datasets[f'{tag}_grlBad'] = {key: np.array(value)[grlBad_mask] for key, value in data_dict.items()}
        ZB_datasets[f'{tag}_grl_nb'] = {key: np.array(value)[grl_nb_mask] for key, value in data_dict.items()}

        del ZB_datasets[tag]

    # Now combine all the tags from each set
    good_tags = [tag for tag in ZB_datasets.keys() if 'grlGood' in tag]
    bad_tags = [tag for tag in ZB_datasets.keys() if 'grlBad' in tag]
    nb_tags = [tag for tag in ZB_datasets.keys() if 'grl_nb' in tag]

    ZB_datasets = combine_data(ZB_datasets, tags_to_combine=good_tags, new_tag='ZB_grlGood')
    ZB_datasets = combine_data(ZB_datasets, tags_to_combine=bad_tags, new_tag='ZB_grlBad')
    ZB_datasets = combine_data(ZB_datasets, tags_to_combine=nb_tags, new_tag='ZB_grl_nb')

    # Print for sanity check
    for tag, data_dict in ZB_datasets.items():
        print(f'{tag}:')
        for key, value in data_dict.items():
            print(f'{key}: {value}')
        print('\n')

    # Plot the ZB scores
    bins = np.linspace(0, 20, 35)
    plt.figure(figsize=(10, 6))
    for tag in ['ZB_grlGood', 'ZB_grlBad', 'ZB_grl_nb']:
        data_dict = ZB_datasets[tag]
        plt.hist(data_dict['HLT_AD_scores'], bins=bins, density=True, histtype='step', linewidth=2.5, fill=False, label=tag)
        
    plt.title(f'HLTAD Score Distributions for all ZeroBias events', fontsize=14)
    plt.xlabel('HLTAD Scores', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
    
    
# -----------------------------------------------------------------------------------------
def anomalous_event_display(datasets: dict, save_path: str, save_name: str):

    # Data
    # jets = [
    #     (159.4, 2.45, 0.74),  # pt, eta, phi
    #     (158.4, 0.95, 4.07),
    #     (16.6, -2.6, 2.65),
    #     (15.2, -1.05, 2.70)
    # ]

    # taus = [
    #     (48.8, 2.45, 0.74),
    #     (43.3, 1.05, 4.17),
    #     (21.6, 0.75, 4.07),
    #     (8.4, -1.95, 1.82)
    # ]
    E_or_F_mask = np.array(['E' in label for label in datasets['EB_test']['region_labels']]) | np.array(['F' in label for label in datasets['EB_test']['region_labels']])
    anom_idx = np.argmax(datasets['EB_test']['HLT_AD_scores'][E_or_F_mask])
    event = datasets['EB_test']['HLT_data'][E_or_F_mask][anom_idx].reshape(16, 3)
    jets = event[0:6]
    electrons = event[6:9]
    muons = event[9:12]
    photons = event[12:15]
    met_pt = event[15, 0]
    met_phi = event[15, 2]

    def eta_to_theta(eta):
        return 2 * np.arctan(np.exp(-eta))

    def draw_cone(ax, pt, eta, phi, R=0.4):
        # Logarithmic scaling to make smaller jets visible
        scale = (0.3 + 0.7 * np.log10(1 + pt)/np.log10(1 + max(j[0] for j in jets)))*3
        
        t = np.linspace(0, 2*np.pi, 50)
        eta_circle = eta + R * np.cos(t)
        phi_circle = phi + R * np.sin(t)
        
        theta_circle = eta_to_theta(eta_circle)
        x = scale * np.sin(theta_circle) * np.cos(phi_circle)
        y = scale * np.sin(theta_circle) * np.sin(phi_circle)
        z = scale * np.cos(theta_circle)
        
        for i in range(len(t)):
            ax.plot([0, x[i]], [0, y[i]], [0, z[i]], 'b-', alpha=0.3)
        
        ax.plot(x, y, z, 'b-', alpha=0.8)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot detector cylinder
    R = 1
    z_points = np.linspace(-4, 4, 100)
    phi_points = np.linspace(0, 2*np.pi, 100)
    Z, P = np.meshgrid(z_points, phi_points)
    X = R * np.cos(P)
    Y = R * np.sin(P)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    # Plot jets
    for pt, eta, phi in jets:
        draw_cone(ax, pt, eta, phi)

    # Plot electrons as rays
    max_electron_pt = max(t[0] for t in electrons)
    for pt, eta, phi in electrons:
        # Logarithmic scaling for taus
        scale = 0.3 + 0.7 * np.log10(1 + pt)/np.log10(1 + max_electron_pt)
        theta = eta_to_theta(eta)
        x = scale * np.sin(theta) * np.cos(phi)
        y = scale * np.sin(theta) * np.sin(phi)
        z = scale * np.cos(theta)
        ax.plot([0, x], [0, y], [0, z], 'orange', alpha=0.8, linewidth=2)

    


    # Plot muons as rays
    max_muon_pt = max(t[0] for t in muons)
    for pt, eta, phi in muons:
        # Logarithmic scaling for taus
        scale = 0.3 + 0.7 * np.log10(1 + pt)/np.log10(1 + max_muon_pt)
        theta = eta_to_theta(eta)
        x = scale * np.sin(theta) * np.cos(phi)
        y = scale * np.sin(theta) * np.sin(phi)
        z = scale * np.cos(theta)
        ax.plot([0, x], [0, y], [0, z], 'green', alpha=0.8, linewidth=2)


    # Plot photons as rays
    max_photon_pt = max(t[0] for t in photons)
    for pt, eta, phi in photons:
        # Logarithmic scaling for taus
        scale = 0.3 + 0.7 * np.log10(1 + pt)/np.log10(1 + max_photon_pt)
        theta = eta_to_theta(eta)
        x = scale * np.sin(theta) * np.cos(phi)
        y = scale * np.sin(theta) * np.sin(phi)
        z = scale * np.cos(theta)
        ax.plot([0, x], [0, y], [0, z], 'red', alpha=0.8, linewidth=2)


    # Plot MET as ray in x-y plane
    # Scale MET separately since it's typically smaller
    max_pt = max(max_electron_pt, max_muon_pt, max_photon_pt)
    scale = 0.3 + 0.7 * met_pt/max_pt  # Using tau scale for reference
    x = scale * np.cos(met_phi)
    y = scale * np.sin(met_phi)
    ax.plot([0, x], [0, y], [0, 0], 'k-', alpha=0.8, linewidth=2)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Most Anomalous Event Display')
    ax.view_init(elev=10, azim=-60)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.3, label='Jets (ΔR=0.4)'),
        Patch(facecolor='orange', alpha=0.8, label='Electrons'),
        Patch(facecolor='green', alpha=0.8, label='Muons'),
        Patch(facecolor='red', alpha=0.8, label='Photons'),
        Patch(facecolor='k', alpha=0.8, label='MET')
    ]
    ax.legend(handles=legend_elements)

    ax.set_box_aspect([1,1,1])

    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()

# -----------------------------------------------------------------------------------------
def plot_latent_space(datasets: dict, save_path: str, save_name: str):
    bins = np.linspace(0, 10, 35)

    zero_mask = np.all(datasets['EB_test']['HLT_preprocessed_data'] == 0, axis=1)

    latent_reps = datasets['EB_test']['z'][~zero_mask]

    plt.figure(figsize=(10, 6))
    for i in range(latent_reps.shape[1]):
        plt.hist(latent_reps[:, i], bins=bins, label=f'latent space dim {i}', density=True, histtype='step', linewidth=2.5, fill=False)


    plt.xlabel('Latent Space Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.savefig(f'{save_path}/{save_name}_1.png')
    plt.close()

    bins = np.linspace(0, 1, 35)

    latent_reps = datasets['EB_test']['z']

    plt.figure(figsize=(10, 6))
    for i in range(latent_reps.shape[1]):
        plt.hist(latent_reps[:, i], bins=bins, label=f'latent space dim {i}', density=True, histtype='step', linewidth=2.5, fill=False)


    plt.xlabel('Latent Space Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.savefig(f'{save_path}/{save_name}_2.png')
    plt.close()

def plot_latent_space_3d(datasets: dict, save_path: str, save_name: str):
    """
    Creates a 3D scatter plot of selected latent space dimensions.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colors for different datasets
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    color_idx = 0
    
    # Skip certain tags that we don't want to plot
    skip_tags = ['EB_train', 'EB_val', 'kaito']
    
    # Loop through datasets and plot each one
    for tag, data in datasets.items():
        if tag in skip_tags or 'ZB' in tag:
            continue

        zero_mask = np.all(data['HLT_preprocessed_data'] == 0, axis=1)
        latent_reps = data['z'][~zero_mask]
        
        # Plot using dimensions 0, 2, and 3
        ax.scatter(latent_reps[:10, 0], 
                  latent_reps[:10, 2], 
                  latent_reps[:10, 3],
                  c=colors[color_idx % len(colors)],
                  label=tag,
                  alpha=0.6,
                  s=10)
        
        color_idx += 1

    # Set axis limits from 0 to 20
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 20)

    ax.set_xlabel('Latent Dim 0')
    ax.set_ylabel('Latent Dim 2') 
    ax.set_zlabel('Latent Dim 3')
    ax.set_title('3D Latent Space Visualization')
    
    # Adjust legend position and size
    ax.legend(bbox_to_anchor=(1.15, 0.5), loc='center left', fontsize=10)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/{save_name}_3d.png', bbox_inches='tight', dpi=300)
    plt.close()


# -----------------------------------------------------------------------------------------
def compare_train_test_scores(datasets:dict, save_path:str, save_name:str):

    train_scores = datasets['EB_train']['HLT_AD_scores']
    test_scores = datasets['EB_test']['HLT_AD_scores']

    plt.figure(figsize=(10, 6))
    bins=np.linspace(0, 20, 100)
    plt.hist(train_scores, bins=bins, label=f'train', density=True, histtype='step', linewidth=2.5, fill=False)
    plt.hist(test_scores, bins=bins, label=f'test', density=True, histtype='step', linewidth=2.5, fill=False)


    plt.xlabel('AD Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
    


def plot_scores_per_run(datasets:dict, save_path:str, save_name:str, L1AD_threshold:float, HLTAD_threshold:float):
    if 'EB_all' in datasets.keys():
    
        plt.figure(figsize=(10, 6))
        unique_runs = np.unique(datasets['EB_all']['run_numbers'])
        for run_num in unique_runs:
            run_mask = datasets['EB_all']['run_numbers'] == run_num
            plt.hist(datasets['EB_all']['HLT_AD_scores'][run_mask], bins=np.linspace(0, 20, 35), label=f'run {run_num}', density=True, histtype='step', linewidth=2.5, fill=False)

        plt.xlabel('AD Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.title(f'AD Scores per Run, all events')
        plt.savefig(f'{save_path}/{save_name}_all_events.png')
        plt.close()


        plt.figure(figsize=(10, 6))
        unique_runs = np.unique(datasets['EB_all']['run_numbers'])
        for run_num in unique_runs:
            run_mask = (datasets['EB_all']['run_numbers'] == run_num)
            passL1AD_mask = datasets['EB_all']['topo2A_AD_scores'] >= L1AD_threshold
            plt.hist(datasets['EB_all']['HLT_AD_scores'][run_mask & passL1AD_mask], bins=np.linspace(0, 20, 35), label=f'run {run_num}', density=True, histtype='step', linewidth=2.5, fill=False)

        plt.xlabel('AD Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.title(f'AD Scores per Run, events passing L1AD')
        plt.savefig(f'{save_path}/{save_name}_L1AD.png')
        plt.close()


        plt.figure(figsize=(10, 6))
        unique_runs = np.unique(datasets['EB_all']['run_numbers'])
        for run_num in unique_runs:
            run_mask = (datasets['EB_all']['run_numbers'] == run_num)
            passL1AD_mask = datasets['EB_all']['topo2A_AD_scores'] >= L1AD_threshold
            fail_HLT_mask = datasets['EB_all']['passHLT'] == False
            plt.hist(datasets['EB_all']['HLT_AD_scores'][run_mask & passL1AD_mask & fail_HLT_mask], bins=np.linspace(0, 20, 35), label=f'run {run_num}', density=True, histtype='step', linewidth=2.5, fill=False)

            passL1AD_mask = np.array(datasets['EB_all']['topo2A_AD_scores'][run_mask] >= L1AD_threshold)
            passHLTAD_mask = np.array(datasets['EB_all']['HLT_AD_scores'][run_mask] >= HLTAD_threshold)
            passHLT_mask = np.array(datasets['EB_all']['passHLT'][run_mask])
            print(f'in the plotting fn::: {run_num}:::')
            print(f'raw number of events passing L1AD and HLTAD: {np.sum(passL1AD_mask & passHLTAD_mask)}')
            print(f'raw number of events passing L1AD and HLTAD and not HLT: {np.sum(passL1AD_mask & passHLTAD_mask & ~passHLT_mask)}')

        plt.xlabel('AD Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.title(f'AD Scores per Run, events passing L1AD and failing HLT')
        plt.savefig(f'{save_path}/{save_name}_L1AD_fail_HLT.png')
        plt.close()

        

# -----------------------------------------------------------------------------------------
        
def plot_individual_model_results(datasets: dict, region_counts: dict, seed_scheme, save_path, model_version, L1AD_threshold, L1AD_rate, HLTAD_threshold, target_HLTAD_rate, obj_type='HLT', plot_ZB=False):

    results_dict = {tag : {} for tag in datasets.keys()}

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

    ROC_curve_other_vars_plot(seeded_datasets, save_path=save_path, save_name=f'ROC_curves_{model_version}_{seed_scheme}', HLTAD_threshold=HLTAD_threshold, obj_type=obj_type)
    #calculate_efficiencies_other_vars(datasets: dict, results_dict: dict, save_path: str, save_name: str, HLTAD_threshold, bkg_tag='EB_test', obj_type='HLT', L1AD_rate=1000):
    results_dict = calculate_efficiencies_other_vars(datasets, results_dict)

    # Plot the raw efficiencies
    raw_efficiencies_plot_from_ROC(seeded_signal_efficiencies, save_path=save_path, save_name=f'Efficiencies_ROC_{model_version}_{seed_scheme}')
    signal_efficiencies = raw_efficiencies_plot_from_regions(datasets, save_path=save_path, save_name=f'Efficiencies_region_counts_{model_version}_{seed_scheme}', seed_scheme=seed_scheme)
    for tag, signal_eff in signal_efficiencies.items():
        results_dict[tag]['HLTAD_raw_efficiency'] = signal_eff

    # Plot the raw HLT efficiencies
    HLT_efficiencies = raw_HLT_efficiencies_plot(datasets, save_path=save_path, save_name=f'HLT_raw_efficiencies_{model_version}')
    for tag, HLT_eff in HLT_efficiencies.items():
        results_dict[tag]['HLT_efficiency'] = HLT_eff

    # Plot the efficiency gains
    EoverFplusG = EoverFplusG_plot(datasets, save_path=save_path, save_name=f'EoverFplusG_{model_version}_{seed_scheme}')
    for tag, EoverFplusG_value in EoverFplusG.items():
        results_dict[tag]['HLTAD_efficiency_gain'] = EoverFplusG_value
    efficiency_gain_plot(region_counts, save_path=save_path, save_name=f'Efficiency_gains_{model_version}_{seed_scheme}', target_rate=target_HLTAD_rate)

    # Plot the efficiency vs variable
    efficiency_vs_variable_plot(datasets, save_path=save_path, save_name=f'Efficiency_plot_{model_version}_{seed_scheme}', obj_type=obj_type, seed_scheme=seed_scheme)

    # Plot E over B plots
    EoverB = EoverB_plot(datasets, save_path=save_path, save_name=f'EoverB_{model_version}_{seed_scheme}')
    E_conditional_overB = E_conditional_overB_plot(datasets, save_path=save_path, save_name=f'E_conditional_overB_{model_version}_{seed_scheme}')
    for tag, E_conditional_overB_value in E_conditional_overB.items():
        results_dict[tag]['HLTAD_E_conditional_overB'] = E_conditional_overB_value

    if plot_ZB:
        ZB_scores_plot(datasets, save_path=save_path, save_name=f'ZB_scores_{model_version}_{seed_scheme}')

    energy_scale_plot(datasets, save_path=save_path, save_name=f'energy_scale_{model_version}')

    AD_score_2d_hists(datasets, save_path=save_path, save_name=f'AD_score_2d_hists_{model_version}_{seed_scheme}')

    #compare_train_test_scores(datasets, save_path=save_path, save_name=f'train_test_scores_{model_version}_{seed_scheme}')
    #plot_scores_per_run(datasets, save_path=save_path, save_name=f'scores_per_run_{model_version}_{seed_scheme}', L1AD_threshold=L1AD_threshold, HLTAD_threshold=HLTAD_threshold)

    # anomalous_event_display(datasets, save_path=save_path, save_name=f'anomalous_event_display_{model_version}_{seed_scheme}')
    # plot_latent_space(datasets, save_path=save_path, save_name=f'latent_space_{model_version}_{seed_scheme}')
    # plot_latent_space_3d(datasets, save_path=save_path, save_name=f'latent_space_3d_{model_version}_{seed_scheme}')
    
    return signal_efficiencies, EoverFplusG, EoverB, results_dict


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
            overlap_removal=training_info['overlap_removal'],
            duplicate_removal=training_info['duplicate_removal']
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
        signal_efficiencies, EoverFplusG, EoverB, results_dict = plot_individual_model_results(
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
        signal_efficiencies, EoverFplusG, EoverB, results_dict = plot_individual_model_results(
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
def process_single_model_from_loaded_datasets(datasets: dict, plots_path: str, model_version: int, target_rate=10, obj_type='HLT', L1AD_rate=1000, custom_threshold=None):

    #datasets['EB_482596_test'] = {key: np.array(value)[(datasets['EB_test']['run_numbers'] == 482596).astype(int)] for key, value in datasets['EB_test'].items()}
    #datasets['EB_482596_train'] = {key: np.array(value)[(datasets['EB_train']['run_numbers'] == 482596).astype(int)] for key, value in datasets['EB_train'].items()}
    #datasets['EB_482596_val'] = {key: np.array(value)[(datasets['EB_val']['run_numbers'] == 482596).astype(int)] for key, value in datasets['EB_val'].items()}
    #datasets = combine_data(datasets, tags_to_combine=['EB_482596_test', 'EB_482596_train', 'EB_482596_val'], new_tag='EB_482596', delete_old_tags=False)
    datasets = combine_data(datasets, tags_to_combine=['EB_test', 'EB_train', 'EB_val'], new_tag='EB_all', delete_old_tags=False)

    print('EB_all shape:')
    for key, value in datasets['EB_all'].items():
        print(f'{key}: {value.shape}')
    datasets['EB_482596'] = {key: np.array(value)[(datasets['EB_all']['run_numbers'] == 482596)] for key, value in datasets['EB_all'].items()}



    # Add these print statements
    run_482596_count = np.sum(datasets['EB_all']['run_numbers'] == 482596)
    eb_482596_count = len(datasets['EB_482596']['run_numbers'])
    print(f"Number of run 482596 events in EB_all: {run_482596_count}")
    print(f"Number of events in EB_482596: {eb_482596_count}")


    # decode the region labels
    for tag, data_dict in datasets.items():
        data_dict['region_labels'] = [label.decode('utf-8') for label in data_dict['region_labels']]

    L1AD_threshold, L1AD_pure_rate , L1AD_total_rate = find_threshold(
            scores=datasets['EB_test']['topo2A_AD_scores'],
            weights=datasets['EB_test']['weights'],
            pass_current_trigs=datasets['EB_test']['passL1'],
            target_rate=L1AD_rate,
            incoming_rate=31575960
    )

    print(f'L1AD_threshold: {L1AD_threshold}')
    print(f'L1AD_pure_rate: {L1AD_pure_rate}')
    print(f'L1AD_total_rate: {L1AD_total_rate}')


    # L1Seeded ---------------------------------------------------------
    pass_L1AD_mask = np.array(datasets['EB_test']['topo2A_AD_scores'] >= L1AD_threshold)

    # HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
    #     scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
    #     weights=datasets['EB_test']['weights'][pass_L1AD_mask],
    #     pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
    #     target_rate=target_rate,
    #     incoming_rate=L1AD_total_rate
    # )
    # print(f'l1Seeded:::')
    # print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
    # print(f'HLTAD_total_rate: {HLTAD_total_rate}')
    # print(f'HLTAD_threshold: {HLTAD_threshold}\n')

    for target_rate in [5, 20, 10]: # since the last value is 10Hz, the variables will correspond to 10Hz after the loop

        HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
            scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
            weights=datasets['EB_test']['weights'][pass_L1AD_mask],
            pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
            target_rate=target_rate,
            incoming_rate=L1AD_total_rate
        )
        print(f'l1Seeded::: target rate: {target_rate}')
        print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
        print(f'HLTAD_total_rate: {HLTAD_total_rate}')
        print(f'HLTAD_threshold: {HLTAD_threshold}\n')

    

    region_counts = {tag: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0} for tag in datasets.keys()}
    
    for tag, data_dict in datasets.items():
        for j, label in enumerate(data_dict['region_labels']):
            weight = data_dict['weights'][j]
            for region in label:
                region_counts[tag][region] += weight

    signal_efficiencies, EoverFplusG, EoverB, results_dict = plot_individual_model_results(
        datasets=datasets, 
        region_counts=region_counts, 
        seed_scheme='l1Seeded',
        save_path=plots_path, 
        model_version=model_version, 
        L1AD_threshold=L1AD_threshold, 
        L1AD_rate=L1AD_total_rate, 
        HLTAD_threshold=HLTAD_threshold,
        target_HLTAD_rate=target_rate,
        obj_type=obj_type,
        plot_ZB=False
    )

    # Now save the event numbers and run numbers of the events that passed the HLTAD trigger
    pass_HLTAD_mask = np.array(['E' in label for label in datasets['EB_test']['region_labels']]) | np.array(['F' in label for label in datasets['EB_test']['region_labels']])
    ev_lb_set = set(zip(datasets['EB_test']['event_numbers'][pass_HLTAD_mask], datasets['EB_test']['run_numbers'][pass_HLTAD_mask]))
    # Save the event numbers and run numbers of events that passed HLTAD
    
    with open(f'{plots_path}/passing_events.txt', 'w') as f:
        for event_num, run_num in ev_lb_set:
            f.write(f'{event_num} {run_num}\n')

    if custom_threshold is not None:
        HLTAD_threshold = custom_threshold

        # Re-calculate the pure and total rates with the custom threshold
        _, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
            scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
            weights=datasets['EB_test']['weights'][pass_L1AD_mask],
            pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
            target_rate=HLTAD_threshold+0.0001,
            incoming_rate=L1AD_total_rate
        )
        print(f'l1Seeded::: with custom threshold:')
        print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
        print(f'HLTAD_total_rate: {HLTAD_total_rate}')
        print(f'HLTAD_threshold: {HLTAD_threshold}\n')

        # Now, re-run the analysis with the custom threshold
        os.makedirs(f'{plots_path}/custom_threshold', exist_ok=True)
        region_counts = {tag: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0} for tag in datasets.keys()}
        
        skip_tags = ['EB_train', 'EB_val']
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

        signal_efficiencies, EoverFplusG, EoverB, _ = plot_individual_model_results(
            datasets=datasets, 
            region_counts=region_counts, 
            seed_scheme='l1Seeded',
            save_path=f'{plots_path}/custom_threshold', 
            model_version=model_version, 
            L1AD_threshold=L1AD_threshold, 
            L1AD_rate=L1AD_total_rate, 
            HLTAD_threshold=HLTAD_threshold,
            target_HLTAD_rate=target_rate,
            obj_type=obj_type,
            plot_ZB=False
        )


    print('Now starting rate calculations for all EB events as well as EB run 482596.')

    def calculate_rate_from_threshold(scores, threshold, weights, pass_current_trigs, incoming_rate):
        total_pass_mask = scores >= threshold
        unique_pass_mask = (scores >= threshold) & (~pass_current_trigs)

        total_rate = np.sum(weights[total_pass_mask]) * incoming_rate / np.sum(weights)
        unique_rate = np.sum(weights[unique_pass_mask]) * incoming_rate / np.sum(weights)
        return total_rate, unique_rate

    print(f'testing EB_482596: {len(datasets["EB_482596"]["topo2A_AD_scores"])}')

    print(f'{len(datasets["EB_482596"]["HLT_AD_scores"])}')
    print(f'{len(datasets["EB_482596"]["weights"])}')
    print(f'{len(datasets["EB_482596"]["passHLT"])}')
    passL1AD_mask = np.array(datasets['EB_482596']['topo2A_AD_scores'] >= L1AD_threshold)
    passHLTAD_mask = np.array(datasets['EB_482596']['HLT_AD_scores'] >= HLTAD_threshold)
    passHLT_mask = np.array(datasets['EB_482596']['passHLT'])
    print(f'raw number of events passing L1AD and HLTAD: {np.sum(passL1AD_mask & passHLTAD_mask)}')
    print(f'raw number of events passing L1AD and HLTAD and not HLT: {np.sum(passL1AD_mask & passHLTAD_mask & ~passHLT_mask)}')

    pass_L1AD_mask = np.array(datasets['EB_all']['topo2A_AD_scores'] >= L1AD_threshold)
    total_rate, unique_rate = calculate_rate_from_threshold(
        scores=datasets['EB_all'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
        threshold=HLTAD_threshold,
        weights=datasets['EB_all']['weights'][pass_L1AD_mask],
        pass_current_trigs=datasets['EB_all']['passHLT'][pass_L1AD_mask],
        incoming_rate=L1AD_total_rate
    )

    print(f'EB ALL:')
    print(f'total rate: {total_rate}')
    print(f'unique rate: {unique_rate}')

    pass_L1AD_mask = np.array(datasets['EB_482596']['topo2A_AD_scores'] >= L1AD_threshold)
    total_rate, unique_rate = calculate_rate_from_threshold(
        scores=datasets['EB_482596'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
        threshold=HLTAD_threshold,
        weights=datasets['EB_482596']['weights'][pass_L1AD_mask],
        pass_current_trigs=datasets['EB_482596']['passHLT'][pass_L1AD_mask],
        incoming_rate=L1AD_total_rate
    )

    print(f'EB 482596:')
    print(f'total rate: {total_rate}')
    print(f'unique rate: {unique_rate}')


    # pass_L1AD_mask = np.array(datasets['EB_all']['topo2A_AD_scores'] >= L1AD_threshold)
    # _, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
    #     scores=datasets['EB_all'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
    #     weights=datasets['EB_all']['weights'][pass_L1AD_mask],
    #     pass_current_trigs=datasets['EB_all']['passHLT'][pass_L1AD_mask],
    #     target_rate=HLTAD_threshold+0.0001,
    #     incoming_rate=L1AD_total_rate
    # )


    # print(f'EB ALL:')
    # print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
    # print(f'HLTAD_total_rate: {HLTAD_total_rate}')


    # pass_L1AD_mask = np.array(datasets['EB_482596']['topo2A_AD_scores'] >= L1AD_threshold)
    # _, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
    #     scores=datasets['EB_482596'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
    #     weights=datasets['EB_482596']['weights'][pass_L1AD_mask],
    #     pass_current_trigs=datasets['EB_482596']['passHLT'][pass_L1AD_mask],
    #     target_rate=HLTAD_threshold+0.0001,
    #     incoming_rate=L1AD_total_rate
    # )
    # print(f'EB 482596:')
    # print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
    # print(f'HLTAD_total_rate: {HLTAD_total_rate}')

    json.dump(results_dict, open(f'{plots_path}/results_dict.json', 'w'))

    return results_dict
    


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

def load_and_inference(training_info: dict, data_info: dict, target_rate: int=10, L1AD_rate: int=1000, obj_type='HLT', save_version:int=0, tags='all', seed_scheme:str='l1Seeded', regions=True):

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

    # Delete datasets that are not in the tags list
    tags_to_delete = [tag for tag in datasets.keys() if tag not in tags]
    if tags != 'all':
        for tag in tags_to_delete:
                del datasets[tag]


    # Load the model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=datasets[list(datasets.keys())[0]]['HLT_data'].shape[1],
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
        overlap_removal=training_info['overlap_removal'],
        duplicate_removal=training_info['duplicate_removal']
    )

    # Pass the data through the model
    

    for tag in datasets.keys():
        data_dict = datasets[tag]

        # Preprocess the data
        data_dict[f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(data_dict[f'{obj_type}_data'], verbose=0, batch_size=8)

        # Calculate the latent representations
        data_dict['z'] = HLT_encoder.predict(data_dict[f'{obj_type}_preprocessed_data'], batch_size=8, verbose=0)

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

        for target_rate in [5, 20, 10]: # since the last value is 10Hz, the variables will correspond to 10Hz after the loop

            HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
                scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
                weights=datasets['EB_test']['weights'][pass_L1AD_mask],
                pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
                target_rate=target_rate,
                incoming_rate=L1AD_total_rate
            )
            print(f'l1Seeded::: target rate: {target_rate}')
            print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
            print(f'HLTAD_total_rate: {HLTAD_total_rate}')
            print(f'HLTAD_threshold: {HLTAD_threshold}\n')

        # Initialize the region counts for each tag
        region_counts = {tag: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0} for tag in datasets.keys()}
        
        # Loop over each tag
        if regions:
            for tag in datasets.keys():
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
            
        
    return datasets, region_counts


def load_and_print_thresholds(training_info: dict, data_info: dict, target_rate: int=10, L1AD_rate: int=1000, obj_type='HLT', save_version:int=0, tags='all', seed_scheme:str='l1Seeded', regions=True):

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

    # Delete datasets that are not in the tags list
    tags_to_delete = [tag for tag in datasets.keys() if tag not in tags]
    if tags != 'all':
        for tag in tags_to_delete:
                del datasets[tag]


    # Load the model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=datasets[list(datasets.keys())[0]]['HLT_data'].shape[1],
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
        overlap_removal=training_info['overlap_removal'],
        duplicate_removal=training_info['duplicate_removal']
    )

    # Pass the data through the model
    

    for tag in datasets.keys():
        if tag != 'EB_test':
            continue
        
        data_dict = datasets[tag]

        # Preprocess the data
        data_dict[f'{obj_type}_preprocessed_data'] = HLT_preprocessing_model.predict(data_dict[f'{obj_type}_data'], verbose=0, batch_size=8)

        # Calculate the latent representations
        data_dict['z'] = HLT_encoder.predict(data_dict[f'{obj_type}_preprocessed_data'], batch_size=8, verbose=0)

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

        for target_rate in [5, 20, 10]: # since the last value is 10Hz, the variables will correspond to 10Hz after the loop

            HLTAD_threshold, HLTAD_pure_rate, HLTAD_total_rate = find_threshold(
                scores=datasets['EB_test'][f'{obj_type}_AD_scores'][pass_L1AD_mask],
                weights=datasets['EB_test']['weights'][pass_L1AD_mask],
                pass_current_trigs=datasets['EB_test']['passHLT'][pass_L1AD_mask],
                target_rate=target_rate,
                incoming_rate=L1AD_total_rate
            )
            print(f'l1Seeded::: target rate: {target_rate}')
            print(f'HLTAD_pure_rate: {HLTAD_pure_rate}')
            print(f'HLTAD_total_rate: {HLTAD_total_rate}')
            print(f'HLTAD_threshold: {HLTAD_threshold}\n')

# -----------------------------------------------------------------------------------------

def create_wrapper_model(preprocessing_model, mse_model):
    raw_inputs = tf.keras.Input(shape=preprocessing_model.input_shape[1:])
    preprocessed_data = preprocessing_model(raw_inputs)
    ad_scores = mse_model(preprocessed_data)
    wrapper_model = tf.keras.Model(inputs=raw_inputs, outputs=ad_scores)

    return wrapper_model

# -----------------------------------------------------------------------------------------
def convert_to_onnx(training_info, data_info, model_version, save_dir, opset=13, input_dim=48, obj_type='HLT'):
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    
    # Load the model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=input_dim,
        pt_thresholds=data_info['pt_thresholds'],
        pt_scale_factor=data_info['pt_scale_factor'],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        saved_model_path=save_path,
        save_version=model_version,
        obj_type=obj_type,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal'],
        duplicate_removal=training_info['duplicate_removal']
    )
    
    # Set models to inference mode
    HLT_preprocessing_model.trainable = False
    HLT_MSE_AE.trainable = False
    
    # Create wrapper model
    wrapper_model = create_wrapper_model(HLT_preprocessing_model, HLT_MSE_AE)
    wrapper_model.trainable = False
    
    # Print model summary for debugging
    wrapper_model.summary()
    
    # Get the exact input shape without batch dimension
    input_shape = wrapper_model.input_shape[1:]
    
    # Convert to ONNX with explicit shape specification
    spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        wrapper_model, 
        opset=opset, 
        input_signature=spec,
        target=["onnxruntime"]
    )
    
    # Save ONNX model
    onnx_file_path = f"{save_dir}/folded_MSE_AE_{model_version}.onnx"
    onnx.save(onnx_model, onnx_file_path)
    print(f"ONNX model saved to: {onnx_file_path}")
    
    return onnx_file_path

# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def compare_tf_with_onnx(datasets: dict, training_info: dict, data_info, model_version, onnx_path, obj_type='HLT', tags='all'):
    """
    Compares the outputs of TensorFlow and ONNX models with detailed diagnostics.
    """
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    
    # Load the TensorFlow model
    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=48,
        pt_thresholds=data_info['pt_thresholds'],
        pt_scale_factor=data_info['pt_scale_factor'],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        saved_model_path=save_path,
        save_version=model_version,
        obj_type=obj_type,
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal'],
        duplicate_removal=training_info['duplicate_removal']
    )
    
    # Create the wrapper model for direct comparison
    wrapper_model = create_wrapper_model(HLT_preprocessing_model, HLT_MSE_AE)
    wrapper_model.trainable = False
    
    # Load the ONNX model
    onnx_path = f"{onnx_path}/folded_MSE_AE_{model_version}.onnx"
    HLT_onnx_session = rt.InferenceSession(onnx_path)
    
    # Determine which tags to process
    skip_tags = ['EB_train', 'EB_val']
    if tags == 'all':
        tags_to_compare = [tag for tag in datasets.keys() if tag not in skip_tags]
    else:
        tags_to_compare = [tag for tag in tags if tag not in skip_tags]
    
    # Dictionary to store comparison statistics
    comparison_stats = {}
    
    for tag in tags_to_compare:
        data_dict = datasets[tag]
        
        # Get raw input data
        raw_input = data_dict[f'{obj_type}_data'].astype(np.float32)
        
        # Run TensorFlow inference (direct through wrapper model)
        tf_scores = wrapper_model.predict(raw_input, verbose=0)
        
        # Run ONNX inference
        onnx_inputs = {HLT_onnx_session.get_inputs()[0].name: raw_input}
        onnx_scores = HLT_onnx_session.run(None, onnx_inputs)[0]
        
        # Store results in dataset
        data_dict[f'{obj_type}_AD_scores'] = tf_scores
        data_dict[f'ONNX_{obj_type}_AD_scores'] = onnx_scores
        
        # Calculate differences
        abs_diff = np.abs(tf_scores - onnx_scores)
        rel_diff = abs_diff / (np.abs(tf_scores) + 1e-10)  # Avoid division by zero
        
        # Store statistics
        comparison_stats[tag] = {
            'max_abs_diff': np.max(abs_diff),
            'mean_abs_diff': np.mean(abs_diff),
            'median_abs_diff': np.median(abs_diff),
            'max_rel_diff': np.max(rel_diff),
            'mean_rel_diff': np.mean(rel_diff),
            'median_rel_diff': np.median(rel_diff),
            'num_significant_diffs': np.sum(abs_diff > 1e-4),
            'total_samples': len(raw_input)
        }
        
        # Find indices of largest differences for inspection
        top_diff_indices = np.argsort(abs_diff.flatten())[-5:]  # Top 5 differences
        
        print(f"\nComparison for {tag}:")
        print(f"Max absolute difference: {comparison_stats[tag]['max_abs_diff']}")
        print(f"Mean absolute difference: {comparison_stats[tag]['mean_abs_diff']}")
        print(f"Number of significant differences: {comparison_stats[tag]['num_significant_diffs']} out of {comparison_stats[tag]['total_samples']}")
        
        print("\nSample of largest differences:")
        for idx in top_diff_indices:
            print(f"Index {idx}: TF={tf_scores.flatten()[idx]}, ONNX={onnx_scores.flatten()[idx]}, Diff={abs_diff.flatten()[idx]}")
    
    return datasets, comparison_stats
# -----------------------------------------------------------------------------------------
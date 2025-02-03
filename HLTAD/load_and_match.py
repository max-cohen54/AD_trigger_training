import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import tarfile
from tensorflow.keras.models import load_model
from qkeras import QConv2D, QBatchNormalization
from qkeras import QDense, QActivation, QDenseBatchnorm, quantized_relu, quantized_bits
from qkeras.utils import _add_supported_quantized_objects
import math


# Infrastructure for loading topo2A data: ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def has_duplicates(arr):
    _, counts = np.unique(arr, return_counts=True)
    return np.any(counts > 1)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def load_and_preprocess_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        # Define object counts
        nmuon, nSRjet, netau = 4, 6, 4

        def fix_phi_range(phi_data):
            # Map values to [0,2pi] range
            phi_fixed = phi_data % (2*math.pi)
            # Scale to [0,128) range and round to nearest integer
            phi_scaled = np.round(phi_fixed * 128/(2*math.pi))
            
            # Take modulo 128 to handle edge cases
            phi_binned = phi_scaled % 128
            return phi_binned
        
        # Load and reshape datasets with scaling
        def load_and_scale(dataset, n_objects, scale_factor=10, eta_factor=40, phi_factor=128/(2*math.pi)):
            data = hf[dataset][:, 0:n_objects, :]
            data[:, :, 0] *= scale_factor  # Scale the pT value
            data[:, :, 1] *= eta_factor  # Scale the angle value
            data[:, :, 2] = fix_phi_range(data[:, :, 2])
            return data.reshape(-1, 3 * n_objects)

        L1_jFexSR_jets = load_and_scale('L1_jFexSR_jets', nSRjet)
        L1_muons = load_and_scale('L1_muons', nmuon, scale_factor=10000)
        L1_eFex_taus = load_and_scale('L1_eFex_taus', netau)

        # Load and process MET
        L1_MET = hf['L1_MET'][:]
        L1_MET[:, 0] *= 40
        L1_MET[:, 2] = fix_phi_range(L1_MET[:, 2])
        
        L1_MET_fixed = np.zeros((L1_MET.shape[0], 2))
        L1_MET_fixed[:, 0] = L1_MET[:, 0]
        L1_MET_fixed[:, 1] = L1_MET[:, 2]
        L1_MET = L1_MET_fixed

        L1_mu = hf['mu'][:]
        valid_events_mask = L1_mu != 0
        
        EB_weights = hf["EB_weights"][:]
        event_number = hf["event_number"][:]
        run_number = hf["run_number"][:]

        weighted_mu_zero_fraction = np.sum(EB_weights[L1_mu == 0]) / np.sum(EB_weights)
        print(f"Weighted fraction of events with mu=0: {weighted_mu_zero_fraction:.4f}")

        pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
                
        # Combine arrays into Topo 2A
        Topo_2A = np.concatenate([L1_jFexSR_jets, L1_eFex_taus, L1_muons, L1_MET], axis=1)
        print("Shape of Topo_2A:", Topo_2A.shape)

        Topo_2A = Topo_2A[valid_events_mask]
        pass_L1_unprescaled = pass_L1_unprescaled[valid_events_mask]
        L1_mu = L1_mu[valid_events_mask]
        EB_weights = EB_weights[valid_events_mask]
        event_number = event_number[valid_events_mask]
        run_number = run_number[valid_events_mask]

        return Topo_2A, pass_L1_unprescaled, L1_mu, EB_weights, event_number, run_number
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def load_and_preprocess_normal_multiple_files(file_paths):
    # Initialize lists to store data from each file
    all_Topo_2A = []
    all_pass_L1 = []
    all_mu = []
    all_weights = []
    all_event_number = []
    all_run_number = []

    
    # Load data from each file
    for file_path in file_paths:
        Topo_2A, pass_L1_unprescaled, L1_mu, L1_weights, L1_event_number, L1_run_number = load_and_preprocess_data(file_path)
        if np.sum(pass_L1_unprescaled) == 0:
            print("no L1 pass events!!!")
        all_Topo_2A.append(Topo_2A)
        all_pass_L1.append(pass_L1_unprescaled)
        all_mu.append(L1_mu)
        all_weights.append(L1_weights)
        all_event_number.append(L1_event_number)
        all_run_number.append(L1_run_number)
    

    # Concatenate all files' data
    final_Topo_2A = np.concatenate(all_Topo_2A, axis=0)
    final_pass_L1 = np.concatenate(all_pass_L1, axis=0)
    final_mu = np.concatenate(all_mu, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    final_event_number = np.concatenate(all_event_number, axis=0)
    final_run_number = np.concatenate(all_run_number, axis=0)

    # Get total number of samples
    n_samples = final_Topo_2A.shape[0]
    
    # Set random seed and create shuffle indices
    np.random.seed(42)
    shuffle_idx = np.random.permutation(n_samples)
    
    # Shuffle all arrays using the same indices
    final_Topo_2A = final_Topo_2A[shuffle_idx]
    final_pass_L1 = final_pass_L1[shuffle_idx]
    final_mu = final_mu[shuffle_idx]
    final_weights = final_weights[shuffle_idx]
    final_event_number = final_event_number[shuffle_idx]
    final_run_number = final_run_number[shuffle_idx]


    def fill_median(array):
        for i in range(array.shape[1]):
            array[np.isnan(array[:, i]), i] = 0#median_value
        return array

    final_Topo_2A = fill_median(final_Topo_2A)

    print(f"total of {n_samples} event processed")
    return final_Topo_2A, final_pass_L1, final_mu, final_weights, final_event_number, final_run_number
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def load_and_process_anomalous_data(file_name):
    with h5py.File(file_name, 'r') as hf:
        nmuon, nLRjet, nSRjet, negamma, netau, njtau = 4, 6, 6, 4, 4, 4
        print(hf.keys())
        def fix_phi_range(phi_data):
            # Map values to [0,2pi] range
            phi_fixed = phi_data % (2*math.pi)
            # Scale to [0,128) range and round to nearest integer
            phi_scaled = np.round(phi_fixed * 128/(2*math.pi))
            
            # Take modulo 128 to handle edge cases
            phi_binned = phi_scaled % 128
            return phi_binned

        def load_and_scale(dataset, n_objects, scale_factor=10, eta_factor=40):
            data = hf[dataset][:, 0:n_objects, :]
            data[:, :, 0] *= scale_factor  # Scale the pT value
            data[:, :, 1] *= eta_factor  # Scale the angle value
            data[:, :, 2] = fix_phi_range(data[:, :, 2])
            return data.reshape(-1, 3 * n_objects)

        L1_jFexSR_jets = load_and_scale('L1_jFexSR_jets', nSRjet)
        L1_jFexLR_jets = load_and_scale('L1_jFexLR_jets', nLRjet)
        L1_egammas = load_and_scale('L1_egammas', negamma)
        L1_muons = load_and_scale('L1_muons', nmuon, scale_factor=10000)  # Specific scaling for muons
        L1_eFex_taus = load_and_scale('L1_eFex_taus', netau)
        L1_jFex_taus = load_and_scale('L1_jFex_taus', njtau)
        try:
            event_number = hf["event_number"][:]
            run_number = hf["run_number"][:]
        except:
            event_number = np.zeros(len(L1_jFexSR_jets))
            run_number = np.zeros(len(L1_jFexSR_jets))

        L1_MET = hf['L1_MET'][:]
        L1_MET[:, 0] *= 40
        L1_MET[:, 2] = fix_phi_range(L1_MET[:, 2])

        pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]

        # Reformat L1_MET
        L1_MET_fixed = np.zeros((L1_MET.shape[0], 2))
        L1_MET_fixed[:, 0] = L1_MET[:, 0]
        L1_MET_fixed[:, 1] = L1_MET[:, 2]
        L1_MET = L1_MET_fixed

        # Combine arrays into Topo groups
        Topo_2A = np.concatenate([L1_jFexSR_jets, L1_eFex_taus, L1_muons, L1_MET], axis=1)

        # Handle NaN values
        def fill_median(array):
            for i in range(array.shape[1]):
                median_value = np.nanmedian(array[:, i])
                array[np.isnan(array[:, i]), i] = 0#median_value
            return array

        Topo_2A = fill_median(Topo_2A)

        return Topo_2A, pass_L1_unprescaled, event_number, run_number
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------

def apply_power_of_2_scaling(X):
    result = [8, 6, 5, 7, 5, 5, 6, 5, 5, 5, 4, 4, 5, 4, 3, 4, 3, 3, 7, 5, 5, 6, 5, 5, 4, 4, 4, 3, 3, 4, 5, 4, 5, 3, 3, 4, 1, 2, 2, -1, 0, 0, 8, 5]
    # Apply the scaling using 2 raised to the power of the result
    X_scaled = X / (2.0 ** np.array(result))
    return X_scaled
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def load_l1AD_model():

    # Define the custom objects dictionary for QKeras layers
    custom_objects = {
        'QDense': QDense,
        'QActivation': QActivation,
        'QDenseBatchnorm': QDenseBatchnorm,
        'quantized_relu': quantized_relu,
        'quantized_bits': quantized_bits
    }

    # Load the model with custom objects
    loaded_model = tf.keras.models.load_model(
        '/eos/home-m/mmcohen/ntuples/l1_model_1-20-2025/',
        custom_objects=custom_objects,
        compile=False  # If they only need inference
    )


    # loaded_model = tf.keras.models.load_model(
    #     model_path,
    #     custom_objects=custom_objects,
    #     compile=False  # If they only need inference
    # )
    return loaded_model
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
def split_data_by_run(datasets, tag_to_split):
    run_numbers = np.unique(datasets[tag_to_split]['run_numbers'])

    # Loop through run numbers and split the data
    for run_num in run_numbers:
        run_mask = datasets[tag_to_split]['run_numbers'] == run_num
        datasets[f"EB_{run_num}"] = {key: value[run_mask] for key, value in datasets[tag_to_split].items()}

    # delete the old tag
    del datasets[tag_to_split]

    # Rename the HLT_noalg_L1All run to avoid confusing it for a regular EB run
    if 'EB_475341' in datasets:
        datasets['HLT_noalg_eb_L1All'] = datasets.pop('EB_475341')

    return datasets
# ---------------------------------------------------------------------


def load_and_match(save_path):
    print('Initializing loading sequence...')
    print('Beginning loading of topo2A data')
    # Load Kenny's topo2A data and keep track of (run number, event number) pairs used for training --------
    
    file_paths = [
        '/eos/home-m/mmcohen/ntuples/EB_h5_10-06-2024/EB_473255_0_10-05-2024.h5',
        '/eos/home-m/mmcohen/ntuples/EB_h5_10-06-2024/EB_475321_0_10-05-2024.h5',
        '/eos/home-m/mmcohen/ntuples/EB_h5_10-06-2024/EB_482596_0_10-05-2024.h5'
    ]    
    
    Topo_2A, pass_L1_unprescaled, event_mu, events_weights, event_number, run_number = load_and_preprocess_normal_multiple_files(file_paths)
    
    train_event_id = event_number[0:1000000]
    train_run_id = run_number[0:1000000]

    topo2A_train_eventNums_runNums = np.array(list(zip(train_event_id, train_run_id))) # *important variable*
    # --------------------------------------------------------------------------------------------------

    
    # Load the new topo2A EB data ---------------------------------------------------------------------
    topo2A_datasets = {}

    # Loop over all the EB h5 files in the directory and append to the lists
    base_path = '/eos/home-m/mmcohen/ntuples/EB_h5_10-06-2024/'
    for file_name in os.listdir(base_path):
        if file_name.startswith('N') or file_name.startswith('.'): continue
        file_path = os.path.join(base_path, file_name)
            
        Topo_2A, pass_L1_unprescaled, event_mu, events_weights, event_number, run_number = load_and_preprocess_data(file_path)
        
        topo2A_datasets[file_name] = {
            'data': Topo_2A,
            'run_numbers': run_number,
            'event_numbers': event_number
        }
    # --------------------------------------------------------------------------------------------------

        
    # Load the new topo2A MC data ----------------------------------------------------------------------
    MC_path = '/eos/home-m/mmcohen/ntuples/MC_07-17-2024/'
    for filename in os.listdir(MC_path):
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = filename.split('_')[0]
    
        Topo_2A, pass_L1, event_number, run_number = load_and_process_anomalous_data(MC_path+filename)
    
        topo2A_datasets[dataset_tag] = {
            'data': Topo_2A[0:100000],
            'event_numbers': event_number,
            'run_numbers': run_number
        }
    # --------------------------------------------------------------------------------------------------



    # Load the topo2A MC23e data ----------------------------------------------------------------------
    MC_path = '/eos/home-m/mmcohen/ntuples/MC23e_12-10-2024/'
    for filename in os.listdir(MC_path):
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = 'mc23e_'+(filename.split('_12')[0]).split('C_')[1]
    
        Topo_2A, pass_L1, event_number, run_number = load_and_process_anomalous_data(MC_path+filename)
    
        topo2A_datasets[dataset_tag] = {
            'data': Topo_2A[0:100000],
            'event_numbers': event_number,
            'run_numbers': run_number
        }
    # --------------------------------------------------------------------------------------------------


    # Load the topo2A physMain data ----------------------------------------------------------------------
    physMain_path = '/eos/home-m/mmcohen/ntuples/physMain/'
    for filename in os.listdir(physMain_path):
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = 'physMain_'+(filename.split('_')[1])+(filename.split('_')[2])
    
        Topo_2A, pass_L1, event_number, run_number = load_and_process_anomalous_data(physMain_path+filename)
    
        topo2A_datasets[dataset_tag] = {
            'data': Topo_2A[0:100000],
            'event_numbers': event_number,
            'run_numbers': run_number
        }
    # --------------------------------------------------------------------------------------------------
    

    
    # Combine all the EB data into one sub-dir ---------------------------------------------------------
    tags_to_combine = [tag for tag in topo2A_datasets.keys() if tag.startswith('EB')]
    topo2A_datasets = combine_data(topo2A_datasets, tags_to_combine=tags_to_combine, new_tag='EB')
    # --------------------------------------------------------------------------------------------------

    # Bitshift all the topo2A data ---------------------------------------------------------------------
    for tag, data_dict in topo2A_datasets.items():
        data_dict['data'] = apply_power_of_2_scaling(data_dict['data'])
    # --------------------------------------------------------------------------------------------------


    print('topo2A data loaded.')

    print('Initializing topo2A model...')
    # Load the topo2A network --------------------------------------------------------------------------
    #model_path = '/eos/home-m/mmcohen/ntuples/2A_AE_model_FDL_FixAll_noEmpty_FocusKL_23e_BESTOFLONGRUN.keras'
    model = load_l1AD_model()
    # --------------------------------------------------------------------------------------------------

    

    print('Model initialized. Starting topo2A inference')
    # Run all events through the network ---------------------------------------------------------------

    for tag, data_dict in topo2A_datasets.items():
        predictions = model.predict(data_dict['data'], verbose=0)
        AD_scores = np.mean(np.square(predictions), axis=1)
        topo2A_datasets[tag]['topo2A_AD_scores'] = AD_scores
    # --------------------------------------------------------------------------------------------------
    print('Inference complete.')

    print('Beginning loading of HLT data...')        
    # Load My MC data -------------------------------------------------------------------------------------
    datasets = {}
    data_path = '../../../../ntuples/MC_07-17-2024/'

    for filename in os.listdir(data_path):
    
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = filename.split('_')[0]
        
        with h5py.File(data_path+filename, 'r') as hf:
            HLT_jets = hf['HLT_jets'][:]
            L1_jFexSR_jets = hf['L1_jFexSR_jets'][:]
            L1_jFexLR_jets = hf['L1_jFexLR_jets'][:]
            HLT_electrons = hf['HLT_electrons'][:]
            LRT_electrons = hf['LRT_electrons'][:]
            L1_egammas = hf['L1_egammas'][:]
            HLT_muons = hf['HLT_muons'][:]
            LRT_muons = hf['LRT_muons'][:]
            L1_muons = hf['L1_muons'][:]
            L1_eFex_taus = hf['L1_eFex_taus'][:]
            L1_jFex_taus = hf['L1_jFex_taus'][:]
            HLT_photons = hf['HLT_photons'][:]
            HLT_MET = hf['HLT_MET'][:].reshape(-1, 1, 4)  # Broadcasting MET
            L1_MET = hf['L1_MET'][:].reshape(-1, 1, 3)
            pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
            pass_HLT_unprescaled = hf["pass_HLT_unprescaled"][:]
    
            HLT_objects = np.concatenate([HLT_jets[:, :6, [0, 2, 3]], HLT_electrons[:, :3, [0, 2, 3]], HLT_muons[:, :3, [0, 2, 3]], HLT_photons[:, :3, [0, 2, 3]], HLT_MET[:, :, [0, 2, 3]]], axis=1)
            L1_objects = np.concatenate([L1_jFexSR_jets[:, :6, :], L1_egammas[:, :3, :], L1_muons[:, :3, :], L1_eFex_taus[:, :3, :], L1_MET], axis=1)
            
            datasets[dataset_tag] = {
                'HLT_data': HLT_objects,
                'L1_data': L1_objects,
                'passL1': pass_L1_unprescaled==1,
                'passHLT': pass_HLT_unprescaled==1,
                'weights': np.ones(len(HLT_objects)),
                'topo2A_AD_scores': topo2A_datasets[dataset_tag]['topo2A_AD_scores']
            }
    
            if len(HLT_objects) > 100000:
                datasets[dataset_tag] = {key: value[:100000] for key, value in datasets[dataset_tag].items()}
    # --------------------------------------------------------------------------------------------------



    # Load MC23e data -------------------------------------------------------------------------------------
    data_path = '/eos/home-m/mmcohen/ntuples/MC23e_12-10-2024/'

    for filename in os.listdir(data_path):
    
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = 'mc23e_'+(filename.split('_12')[0]).split('C_')[1]
        
        with h5py.File(data_path+filename, 'r') as hf:
            HLT_jets = hf['HLT_jets'][:]
            L1_jFexSR_jets = hf['L1_jFexSR_jets'][:]
            L1_jFexLR_jets = hf['L1_jFexLR_jets'][:]
            HLT_electrons = hf['HLT_electrons'][:]
            LRT_electrons = hf['LRT_electrons'][:]
            L1_egammas = hf['L1_egammas'][:]
            HLT_muons = hf['HLT_muons'][:]
            LRT_muons = hf['LRT_muons'][:]
            L1_muons = hf['L1_muons'][:]
            L1_eFex_taus = hf['L1_eFex_taus'][:]
            L1_jFex_taus = hf['L1_jFex_taus'][:]
            HLT_photons = hf['HLT_photons'][:]
            HLT_MET = hf['HLT_MET'][:].reshape(-1, 1, 3)  # Broadcasting MET
            L1_MET = hf['L1_MET'][:].reshape(-1, 1, 3)
            pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
            pass_HLT_unprescaled = hf["pass_HLT_unprescaled"][:]
            event_number = hf["event_number"][:]
            run_number = hf["run_number"][:]
            mu = hf["mu"][:]
    
            HLT_objects = np.concatenate([HLT_jets[:, :6, [0, 2, 3]], HLT_electrons[:, :3, :], HLT_muons[:, :3, :], HLT_photons[:, :3, :], HLT_MET], axis=1)
            L1_objects = np.concatenate([L1_jFexSR_jets[:, :6, :], L1_egammas[:, :3, :], L1_muons[:, :3, :], L1_eFex_taus[:, :3, :], L1_MET], axis=1)
            
            datasets[dataset_tag] = {
                'HLT_data': HLT_objects,
                'L1_data': L1_objects,
                'passL1': pass_L1_unprescaled==1,
                'passHLT': pass_HLT_unprescaled==1,
                'weights': np.ones(len(HLT_objects)),
                'topo2A_AD_scores': topo2A_datasets[dataset_tag]['topo2A_AD_scores'],
                'event_numbers': event_number,
                'run_numbers': run_number,
                'pileups': mu
            }
    
            if len(HLT_objects) > 100000:
                datasets[dataset_tag] = {key: value[:100000] for key, value in datasets[dataset_tag].items()}
    # --------------------------------------------------------------------------------------------------

    

    # Load physMain data -------------------------------------------------------------------------------------
    data_path = '/eos/home-m/mmcohen/ntuples/physMain/'

    for filename in os.listdir(data_path):
    
        if filename.startswith('N') or filename.startswith('.'): continue
    
        dataset_tag = 'physMain_'+(filename.split('_')[1])+(filename.split('_')[2])
        
        with h5py.File(data_path+filename, 'r') as hf:
            HLT_jets = hf['HLT_jets'][:]
            L1_jFexSR_jets = hf['L1_jFexSR_jets'][:]
            L1_jFexLR_jets = hf['L1_jFexLR_jets'][:]
            HLT_electrons = hf['HLT_electrons'][:]
            LRT_electrons = hf['LRT_electrons'][:]
            L1_egammas = hf['L1_egammas'][:]
            HLT_muons = hf['HLT_muons'][:]
            LRT_muons = hf['LRT_muons'][:]
            L1_muons = hf['L1_muons'][:]
            L1_eFex_taus = hf['L1_eFex_taus'][:]
            L1_jFex_taus = hf['L1_jFex_taus'][:]
            HLT_photons = hf['HLT_photons'][:]
            HLT_MET = hf['HLT_MET'][:].reshape(-1, 1, 3)  # Broadcasting MET
            L1_MET = hf['L1_MET'][:].reshape(-1, 1, 3)
            pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
            pass_HLT_unprescaled = hf["pass_HLT_unprescaled"][:]
            event_number = hf["event_number"][:]
            run_number = hf["run_number"][:]
            mu = hf["mu"][:]
    
            HLT_objects = np.concatenate([HLT_jets[:, :6, [0, 2, 3]], HLT_electrons[:, :3, :], HLT_muons[:, :3, :], HLT_photons[:, :3, :], HLT_MET], axis=1)
            L1_objects = np.concatenate([L1_jFexSR_jets[:, :6, :], L1_egammas[:, :3, :], L1_muons[:, :3, :], L1_eFex_taus[:, :3, :], L1_MET], axis=1)
            
            datasets[dataset_tag] = {
                'HLT_data': HLT_objects,
                'L1_data': L1_objects,
                'passL1': pass_L1_unprescaled==1,
                'passHLT': pass_HLT_unprescaled==1,
                'weights': np.ones(len(HLT_objects)),
                'topo2A_AD_scores': topo2A_datasets[dataset_tag]['topo2A_AD_scores'],
                'event_numbers': event_number,
                'run_numbers': run_number,
                'pileups': mu
            }
    
            if len(HLT_objects) > 100000:
                datasets[dataset_tag] = {key: value[:100000] for key, value in datasets[dataset_tag].items()}
    # --------------------------------------------------------------------------------------------------



    # Load my EB data ----------------------------------------------------------------------------------
    base_path = '/eos/home-m/mmcohen/ntuples/EB_h5_10-06-2024/'
    
    # Iterate over all files in the directory
    for file_name in os.listdir(base_path):
        if file_name.startswith('N') or file_name.startswith('.'): continue
        file_path = os.path.join(base_path, file_name)
        
        # Open each h5 file and append data to lists
        with h5py.File(file_path, 'r') as hf:
            HLT_jets = hf['HLT_jets'][:]
            ofl_jets = hf['ofl_jets'][:]
            L1_jFexSR_jets = hf['L1_jFexSR_jets'][:]
            L1_jFexLR_jets = hf['L1_jFexLR_jets'][:]
            HLT_electrons = hf['HLT_electrons'][:]
            LRT_electrons = hf['LRT_electrons'][:]
            ofl_electrons = hf['ofl_electrons'][:]
            L1_egammas = hf['L1_egammas'][:]
            HLT_muons = hf['HLT_muons'][:]
            LRT_muons = hf['LRT_muons'][:]
            ofl_muons = hf['ofl_muons'][:]
            L1_muons = hf['L1_muons'][:]
            L1_eFex_taus = hf['L1_eFex_taus'][:]
            L1_jFex_taus = hf['L1_jFex_taus'][:]
            HLT_photons = hf['HLT_photons'][:]
            ofl_photons = hf['ofl_photons'][:]
            HLT_MET = hf['HLT_MET'][:].reshape(-1, 1, 3)  # Broadcasting MET
            L1_MET = hf['L1_MET'][:].reshape(-1, 1, 3)
            pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
            pass_HLT_unprescaled = hf["pass_HLT_unprescaled"][:]
            EB_weights = hf["EB_weights"][:]
            event_number = hf["event_number"][:]
            run_number = hf["run_number"][:]
            mu = hf["mu"][:]
    
        HLT_objects = np.concatenate([HLT_jets[:, :6, [0, 2, 3]], HLT_electrons[:, :3, :], HLT_muons[:, :3, :], HLT_photons[:, :3, :], HLT_MET], axis=1)
        L1_objects = np.concatenate([L1_jFexSR_jets[:, :6, :], L1_egammas[:, :3, :], L1_muons[:, :3, :], L1_eFex_taus[:, :3, :], L1_MET], axis=1)
        
        datasets[file_name.split('_10')[0]] = {
            'HLT_data': HLT_objects,
            'L1_data': L1_objects,
            'passL1': pass_L1_unprescaled==1,
            'passHLT': pass_HLT_unprescaled==1,
            'weights': EB_weights,
            'event_numbers': event_number,
            'run_numbers': run_number,
            'pileups': mu
        }
    # --------------------------------------------------------------------------------------------------

    
    # Combine all the EB data into one sub-dir ---------------------------------------------------------
    tags_to_combine = [tag for tag in datasets.keys() if tag.startswith('EB')]
    datasets = combine_data(datasets, tags_to_combine=tags_to_combine, new_tag='EB')
    # --------------------------------------------------------------------------------------------------

    # Remove events with no pileup ---------------------------------------------------------------------
    for tag, data_dict in datasets.items():
        if 'pileups' in data_dict:
            pileup_nonzero = data_dict['pileups'] != 0
            datasets[tag] = {key: value[pileup_nonzero] for key, value in data_dict.items()}
    # --------------------------------------------------------------------------------------------------

    print('HLT data successfully loaded.')

    print('Initializing data organization sequence phase 1: matching topo2A AD scores to HLT events')
    # match topo2A_AD_scores to my events using the (run number, event number) pairs -------------------
    datasets['EB']['topo2A_AD_scores'] = np.zeros(len(datasets['EB']['run_numbers']))
    
    topo2A_lookup = {
        (topo_run_num, topo_ev_num): score
        for topo_run_num, topo_ev_num, score in zip(
            topo2A_datasets['EB']['run_numbers'],
            topo2A_datasets['EB']['event_numbers'],
            topo2A_datasets['EB']['topo2A_AD_scores']
        )
    }

    # Populate datasets['EB']['topo2A_AD_scores'] based on the lookup dictionary
    for i_HLT, (HLT_run_num, HLT_ev_num) in enumerate(zip(datasets['EB']['run_numbers'], datasets['EB']['event_numbers'])):
        datasets['EB']['topo2A_AD_scores'][i_HLT] = topo2A_lookup.get((HLT_run_num, HLT_ev_num), 0)
    # --------------------------------------------------------------------------------------------------
    print('phase 1 complete.')


    print('Initializing data organization sequence phase 2: separating events used to train topo2A model')
    # Finally, we can split off the events used to train topo2A model to avoid eval over those events --
    # Convert the important variable to a set for efficient lookup
    topo2A_train_set = set(map(tuple, topo2A_train_eventNums_runNums))
    
    # Initialize empty dictionaries for the new datasets
    datasets['topo2A_train'] = {key: [] for key in datasets['EB']}
    datasets['EB_rest'] = {key: [] for key in datasets['EB']}
    
    # Loop over each event in datasets['EB'] and split based on the condition
    for i, (run_num, event_num) in enumerate(zip(datasets['EB']['run_numbers'], datasets['EB']['event_numbers'])):
        if (event_num, run_num) in topo2A_train_set:
            # If the pair is in topo2A_train_set, add it to datasets['topo2A_train']
            for key in datasets['EB']:
                datasets['topo2A_train'][key].append(datasets['EB'][key][i])
        else:
            # Otherwise, add it to datasets['EB_rest']
            for key in datasets['EB']:
                datasets['EB_rest'][key].append(datasets['EB'][key][i])
    
    # Convert lists back to numpy arrays
    for key in datasets['topo2A_train']:
        datasets['topo2A_train'][key] = np.array(datasets['topo2A_train'][key])
        datasets['EB_rest'][key] = np.array(datasets['EB_rest'][key])

    del datasets['EB'] # Delete the old 'EB' dataset

    # Split the EB_rest into different runs
    datasets = split_data_by_run(datasets, tag_to_split='EB_rest')
    # --------------------------------------------------------------------------------------------------
    print('phase 2 complete.')

    print('Begin data saving sequence...')
    save_subdicts_to_h5(datasets, save_path)

    os.makedirs(save_path + '/topo2A_datasets', exist_ok=True)
    save_subdicts_to_h5(topo2A_datasets, save_path + '/topo2A_datasets')

    print('Data saved successfully.')
    
    print('\n\npowering down... goodbye.')
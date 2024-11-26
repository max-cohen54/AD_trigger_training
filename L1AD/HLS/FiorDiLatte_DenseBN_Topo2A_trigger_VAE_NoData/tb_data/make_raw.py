import h5py
import numpy as np
import os

def process_dataset(data):
    # Create a new array with the same shape as the input data
    processed_data = np.zeros(data.shape, dtype=np.int32)
    
    # Convert specific columns to unsigned int
    uint_columns = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
    for col in uint_columns:
        processed_data[:, col] = data[:, col].astype(np.uint32)
    
    # Convert the rest of the columns to int
    int_columns = [i for i in range(data.shape[1]) if i not in uint_columns]
    for col in int_columns:
        processed_data[:, col] = data[:, col].astype(np.int32)
    
    return processed_data

def save_to_dat(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            f.write(' '.join(map(str, row)) + '\n')

# Use the existing 'raw_data' directory
output_dir = 'raw_data'
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"The directory '{output_dir}' does not exist. Please create it before running this script.")

# Open the input file
with h5py.File('preprocessed_2A_data_raw_final.h5', 'r') as input_file:
    # Process each dataset
    for key in input_file.keys():
        data = input_file[key][:]
        if '_pass_test' in key.lower():
            continue

        # Skip processing for datasets with 'weights' in the name
        if 'weights' in key.lower():
            output_filename = os.path.join(output_dir, f"{key}.dat")
            #save_to_dat(data, output_filename)
            print(f"skip weight data to {output_filename}")
        else:
            # Ensure the dataset has 44 columns
            if data.shape[1] != 44:
                print(f"Warning: Dataset {key} does not have 44 columns. Saving original data.")
                output_filename = os.path.join(output_dir, f"{key}.dat")
                save_to_dat(data, output_filename)
            else:
                processed_data = process_dataset(data)
                output_filename = os.path.join(output_dir, f"{key}.dat")
                save_to_dat(processed_data, output_filename)
                print(f"Processed and saved data to {output_filename}")

print("Data processing complete. Processed data saved as .dat files in the 'raw_data' directory.")

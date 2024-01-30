# AD_trigger_training

## Introduction
This repo is primaily used to train and store models for Penn/SLAC's work on developing an AD trigger. 
There's a directory for trained models; if you add to this, please update the text file with a short description of what/how you trained.

# Setup
We have made ntuples from Enhanced Bias data, including calculation of the EB weights and running the events through a simulated L1 and HLT.
These ntuples can be accessed from `https://cernbox.cern.ch/s/hzwVaCmvNq4xl7A`, and can be read in with python. I recommend using SWAN,
which makes it much easier to code in python, and provides GPUs for training neural nets. The EB data can be read in by doing something like

```
with h5py.File('../ntuples/AOD_EB_ntuples_1-26-2024.h5', 'r') as hf:
    jets = hf['HLT_jets'][:]
    electrons = hf['electrons'][:]
    LRT_electrons = hf['LRT_electrons'][:]
    muons = hf['muons'][:]
    LRT_muons = hf['LRT_muons'][:]
    photons = hf['photons'][:]
    MET = hf['MET'][:].reshape(-1, 1, 4)  # Broadcasting MET
    pass_L1_unprescaled = hf["pass_L1_unprescaled"][:]
    EB_weights = hf["EB_weights"][:]

pass_L1_idxs = (pass_L1_unprescaled == 1) # Indices of events that passed L1
data = np.concatenate([jets[pass_L1_idxs], electrons[pass_L1_idxs], muons[pass_L1_idxs], photons[pass_L1_idxs], MET[pass_L1_idxs]], axis=1)
weights = EB_weights[pass_L1_idxs]
```

where `jets` will have shape `(N, 10, 4)`, which represents N events, keeping the hardest 10 jets, the last index runing over `(pt, E, eta, phi)`.
`electrons`, `muons`, and `photons` will all have shape `(N, 3, 4)`, where we are only keeping the 3 hardest of these objects.

`pass_L1_idxs` is a list of bits 0 or 1 for each event. 1 means that the event passed at least one trigger from a list of the highest rate unprescaled 
L1 triggers. In practice, the events where this value is 1 should make up (with EB weights) a dataset that is representative of 'as seen by HLT' data.

If you are interested in making your own ntuples, refer to `https://gitlab.cern.ch:8443/mmcohen/ntuple-dumper-x-aod-ana-helpers-minimal-example` for
instructions on how to process datasets and create a TTree with the wanted data. One can then use `/data_pipeline/read_EB_tree.ipynb` to access the 
data from the TTree including the simulated trigger decisions, process the EB weight xml files, and write everything to an h5 file.

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


# Information about the Enhanced Bias data
Recent EB data gets repossessed weekly

Weights are calculated for each (good) event

Simulated L1 and HLT are applied and trigger decisions are saved (all with PS=1)

Here’s a recent reprocessing: `https://its.cern.ch/jira/browse/ATR-28661`
you can look at the HLT Reprocessings label to see them all
By clicking on the Panda [task], then scroll down to the bottom Slice outputs: AOD and click the green ‘finished’ we can see the output collection was `data22_13p6TeV.00440499.physics_EnhancedBias.merge.AOD.r15247_r15248_p6016_tid36850978_00`

EB weights are kept in XML files, which can be read in with a short script. 

List of datasets and location of weight XMLs: `https://twiki.cern.ch/twiki/bin/viewauth/Atlas/EnhancedBiasData`

Existing c++ tool to read the weights: `https://acode-browser1.usatlas.bnl.gov/lxr/source/athena/Trigger/TrigCost/EnhancedBiasWeighter/EnhancedBiasWeighter/EnhancedBiasWeighter.h`

It ended up being easier to write our own python script to read in the weights from the XML: `/data_pipeline/EB_weighter.py`

Simulated L1 and HLT trigger decisions can be accessed normally through the TrigDecisionTool (or normally through xAODAnaHelpers).


EB data (with the weights) should be representative of the “as seen by L1” data, and can therefore be used for L1 studies.


In order to obtain “as seen by HLT” data from the EB, we compiled a large list of PS=1 L1 physics triggers with the largest rates: `https://atlas-runquery.cern.ch/query.py?q=find+r+data22_13p6TeV.periodF+%2F+show+trigkeys`
Click ‘rates’ to see rates of the triggers
Then we made a dataset only with events that passed one of these triggers.

Later this year, some folks are planning to collect a separate dataset which is streamed only based on HLT_noalg_L1All, which would directly be the “as seen by HLT” data.



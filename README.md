# AD_trigger_training

## Introduction
This repo is primaily used to train and store models for work on developing an AD trigger originating from Penn/Slac, with now more collaborators. 
There's a directory for trained models; if you add to this, please update the text file with a short description of what/how you trained.
A list of h5 data files can be found in `n_tuple_locations_list.txt`, see this for information about the most recent data we've been using for training.

If one is interested in making their own ntuples, please refer to https://github.com/max-cohen54/ntuple-dumper/tree/main.

# Setup
We have made ntuples from Enhanced Bias data, including calculation of the EB weights and running the events through a simulated L1 and HLT.
These ntuples are stored in `/eos/home-m/mmcohen/ntuples/`. Information about the most recent ntuples can be found in `n_tuple_locations_list.txt`.

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



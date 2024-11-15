# HLT Algorithm Development

This directory is used for training and running evals for the HLT anomaly detection network. `load_and_match.py` and `ensembler_functions.py` contain useful functions for these trainings, and `example_training.ipynb` goes through an example of how to use them. There is also a function which converts the tf/keras model to ONNX after training, which is also shown in the example notebook. One such ONNX model can be found in `./onnx_models`.

Additionally, `training_documentation.txt` holds information about the training data and network parameters for each training run.

## Additional details
We are still testing a few different input schemes for the HLTAD network. One such scheme involves seeding with the L1AD network; accordingly, it is necessary to attain L1AD scores for each event, in order to determine which events will actually be seen by the HLT network. `load_and_match.py` does this. More specifically, it:
1. Loads the data for the L1AD network
2. Loads the L1AD model
3. calculates L1AD scores for each event
4. Loads the data for the HLTAD network
5. Matches L1AD scores for each HLTAD event.

all of this is done while also keeping track of which events were used to train the L1AD network (we don't want to accidentally run evals over those events).

### Loading the data

First, we'll have to load the L1AD model and calculate L1AD scores. The newest version of the L1AD model requires a few modifications to qkeras; it might thus be easiest to set up the conda environment in `l1AD_softwarew_env.yml`, and edit qkeras locally. After setting up the environment, navigate to the qkeras folder, and follow `https://github.com/google/qkeras/pull/74/files`. You only need to modify `qkeras/__init__.py`, add `qkeras/qdense_batchnorm.py`, and modify `qkeras/utils.py`.

Once this is done, you can run `run_lam.py` with the appropriate paths, which will load the L1AD model, and run inference over all events to get L1AD scores. It also saves these results to the specified path, so it only has to be run once.

### Training and Evals

`ensembler_functions.py` contains the actual infrastructure of training and running evals for the HLTAD network. First, data is loaded (from the output of `run_lam.py`). Next, one can train multiple networks with the same parameters (to ensure that results are stable). Finally, one can easily run evals. An example of this can be found in `example_training.ipynb`.
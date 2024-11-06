# HLT Algorithm Development

This directory is used for training and running evals for the HLT anomaly detection network. `load_and_match.py` and `ensembler_functions.py` contain useful functions for these trainings, and `example_training.ipynb` goes through an example of how to use them. There is also a function which converts the tf/keras model to ONNX after training, which is also shown in the example notebook. One such ONNX model can be found in `./onnx_models`.

## Additional details
We are still testing a few different input schemes for the HLTAD network. One such scheme involves seeding with the L1AD network; accordingly, it is necessary to attain L1AD scores for each event, in order to determine which events will actually be seen by the HLT network. `load_and_match.py` does this. More specifically, it:
1. Loads the data for the L1AD network
2. Loads the L1AD model
3. calculates L1AD scores for each event
4. Loads the data for the HLTAD network
5. Matches L1AD scores for each HLTAD event.

all of this is done while also keeping track of which events were used to train the L1AD network (we don't want to accidentally run evals over those events).

All of this can be done with
```
import load_and_match as lam
lam.load_and_match(save_path)
```
where `save_path` is the path in which the output of this is saved (such that it only has to be run once).

`ensembler_functions.py` contains the actual infrastructure of training and running evals for the HLTAD network. First, data is loaded (from the output of load_and_match). Next, one can train multiple networks with the same parameters (to ensure that results are stable). Finally, one can easily run evals. All of this can be done by running 

```
import ensembler_functions as ef
L1AD_rate = 1000
target_rate = 10
data_info = {
    "train_data_scheme": "topo2A_train+overlap", 
    "pt_normalization_type": "StandardScaler", 
    "L1AD_rate": 1000
}

training_info = {
    "save_path": "./trained_models/multiple_trainings/trial_8", 
    "dropout_p": 0.1, 
    "L2_reg_coupling": 0.01, 
    "latent_dim": 4, 
    "large_network": True, 
    "num_trainings": 10,
    "training_weights": True
}

datasets, data_info = ef.load_and_preprocess(**data_info)
training_info, data_info = ef.train_multiple_models(datasets, data_info, **training_info)
```
which trains ten networks, followed by

```
ef.process_multiple_models(
    training_info=training_info,
    data_info=data_info,
    plots_path=training_info['save_path']+'/plots',
    target_rate=target_rate,
    L1AD_rate=L1AD_rate
)
```

which runs evals, and writes many plots to the specified path.
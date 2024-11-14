import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import h5py
import os
import struct

# Set the base directory
base_dir = "./"

# Read log files
br_loss = np.array([np.mean([float(x)**2 for x in line.strip().split()]) 
                    for line in open(os.path.join(base_dir, "Topo_2A_L1failed_csim.log"))])

signal_loss = [
    np.array([np.mean([float(x)**2 for x in line.strip().split()]) 
              for line in open(os.path.join(base_dir, f"Topo_2A_{channel}_pure_csim.log"))])
    for channel in ["A14", "HAHMggf", "qqa", "Zprime", "ZZ4lep", "HHbbtt"]
]

# Read weights for normal events
with h5py.File("preprocessed_2A_data_raw_final.h5", "r") as hf:
    br_weights = hf['Topo_EB_L1failed_weights'][:]

# Set up variables
labels = ["HNL", "HAHMggf", "qqa", "Zprime", "ZZ4lep", "HHbbtt"]
title = "Anomaly Detection"
saveplots = True

target_fpr = 2.5e-5/(1-0.18456571428571428)
tpr_at_target = []
thresholds_at_target = []

plt.figure(figsize=(10, 8))
plt.plot((1-0.18456571428571428)*40000000*np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal")

br_loss = np.atleast_1d(br_loss)
for j, batch_loss in enumerate(signal_loss):
    sig_w = np.ones(len(batch_loss))  # All anomaly event weights are 1
    br_w = br_weights
    weights = np.concatenate((br_w, sig_w))
    truth = np.concatenate([np.zeros(len(br_loss)), np.ones(len(batch_loss))])
    ROC_data = np.concatenate((br_loss, batch_loss))
    fpr, tpr, thresholds = sk.roc_curve(truth, ROC_data, sample_weight=weights)
    auc = sk.roc_auc_score(truth, ROC_data, sample_weight=weights)
    plt.plot(fpr*(1-0.18456571428571428)*40000000, tpr, label=f"{labels[j]}: AUC = {auc:.3f}")
    idx = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target.append(tpr[idx])
    thresholds_at_target.append(thresholds[idx])

plt.xlabel("Trigger frequency [Hz]")
plt.xlim([1, (1-0.18456571428571428)*40000000])
plt.ylim([5e-8, 1.5])
plt.ylabel("True Positive Rate")
plt.xscale('log')
plt.yscale('log')
plt.title(f"{title} ROC")
plt.vlines(1e3, 0, 1, colors="r", linestyles="dashed")
plt.legend(loc="lower right")
plt.grid(True, which="both", ls="-", alpha=0.2)

if saveplots:
    plt.savefig(f"FDL_ROC.png", format="png", bbox_inches="tight", dpi=1200)
plt.show()

print(f"\nTPR at FPR = {target_fpr} for each channel:")
for label, tpr, threshold in zip(labels, tpr_at_target, thresholds_at_target):
    print(f"{label}: {100*tpr:.6f}%, Threshold = {threshold:.30f}")
    print(''.join(f'{byte:08b}' for byte in struct.pack('!d', threshold)))

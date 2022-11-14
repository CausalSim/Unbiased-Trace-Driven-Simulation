# Heterogeneous load balancing simulation with CausalSim

To reproduce the results in the paper, we need to:

0. Install required python packages
1. Create the dataset (~XGB).
2. Train CausalSim models
3. Carry out counterfactual simulations with these models
4. Choose the correct hyperparameter
5. Plot full-scale simulation results

---
## 0. Python packages
We use Python (3.8 tested) for all experiments. Install the following packages via `pip` or `conda` :
```
numpy, pandas, tqdm, matplotlib, tensorboard
```
Install PyTorch according to the website [instructions](https://pytorch.org).

---
## 1. Preparing the dataset

First, create a directory as a workspace for datasets, models, simulations and plots. We'll call this directory CAUSALSIM_DIR.
Next, run the following command:
```
python3 create_dataset.py --dir CAUSALSIM_DIR
```
This script will create a synthetic dataset for heterogeneous load balancing and save it in `CAUSALSIM_DIR`.

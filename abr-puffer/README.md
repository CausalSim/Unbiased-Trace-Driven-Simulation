# Real-world Adaptive BitRate (ABR) simulation with CausalSim

To reproduce the results in the paper, we need to:
0. Install required python packages
1. Download and prepare the dataset (~40GB).
2. Train CausalSim models
3. Carry out counterfactual simulations with these models
4. Choose the correct hyperparameter
5. Plot full-scale simulation results

---
## 0. Python packages
We use Python (3.8 tested) for all experiments. Install the following packages via `pip` or `conda` :
```
numpy, pandas, tqdm, matplotlib, scikit-learn
```
Install PyTorch according to the website [instructions](https://pytorch.org).

---
## 1. Preparing the dataset

First, create a directory as a workspace for datasets, models, simulations and plots. We'll call this directory CAUSALSIM_DIR.
Next, run the following command:
```
python3 data_preparation/create_dataset.py --dir CAUSALSIM_DIR
```
This script will download stream logs from [the puffer website](https://puffer.stanford.edu). It will then filter them 
according to the [puffer-statistics](https://github.com/StanfordSNR/puffer-statistics) definition of `slow streams`. 
The dataset is saved in `CAUSALSIM_DIR/cooked`.

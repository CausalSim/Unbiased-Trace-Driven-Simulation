# Synthetic Adaptive BitRate (ABR) simulation with CausalSim

To reproduce the results in the paper, we need to:

0. Install required python packages
1. Create the dataset and ExpertSim trajectories (~14GB).
2. Train, Infer and Tune CausalSim models, and plot results.

---
## 0. Python packages
We use Python (3.8 tested) for all experiments. Install the following packages via `pip` or `conda` :
```
numpy, scipy, tqdm, matplotlib, scikit-learn, cython
```
Install PyTorch according to the website [instructions](https://pytorch.org).

---
## 1. Preparing the dataset

Begin by compiling a fast cython-based MPC implementation:
```
cd cpolicies
make all
cd ..
```

Then, create a directory as a workspace for datasets, models, simulations and plots. We'll call this directory CAUSALSIM_DIR.
Next, run the following command:
```
python3 create_dataset_and_expertsim.py --dir CAUSALSIM_DIR
```
This script will create traces and stream logs for a live-streaming session (maximum buffer is limited to 10 seconds), saved in `CAUSALSIM_DIR`.
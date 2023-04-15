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

## 2. Train/Infer/Tune/Plot

For the synthetic ABR experiment, we generate counterfactual predictions for a synthetic ABR environment for one of the following policies:

1. **BBA**.
2. **BBAMIX-x1-50**.
3. **BBAMIX-x2-50**.
4. **MPC**.
5. **Random**.
6. **BOLA**.
7. **Rate Based**.
8. **Optimistic Rate Based**.
9. **Pessimistic Rate Based**.

To run the experiment and generate the counterfactuals of policy `POLICY_NAME` (while leaving the policy out of the training data), run:


```

python3 main.py --policy_out POLICY_NAME --dir CAUSALSIM_DIR --slsim_loss  mse_loss

```

Where `POLICY_NAME` can be any one of {
    "BBA",
    "BBAMIX-x1-50",
    "BBAMIX-x2-50",
    "MPC",
    "Random",
    "BOLA",
    "Rate Based",
    "Optimistic Rate Based",
    "Pessimistic Rate Based"
}.

and slsim loss should be one of the three losses `mse_loss`, `l1_loss`, or `huber_loss`.

This will produce a plots (saved in `figures/`) showing the MSE of estimating the counterfactual buffer occupancy for both CausalSim and SLSim.
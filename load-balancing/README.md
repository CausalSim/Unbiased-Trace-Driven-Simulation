# Heterogeneous load balancing simulation with CausalSim

To reproduce the results in the paper, we need to:

0. Install required python packages
1. Create the dataset (~2GB).
2. Train, Infer and Tune CausalSim models, and plot results.

---
## 0. Python packages
We use Python (3.8 tested) for all experiments. Install the following packages via `pip` or `conda` :
```
numpy, tqdm, matplotlib, tensorboard, scikit-learn
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

## 2. Train/Infer/Tune/Plot

For the load balancing experiment, we generate counterfactual predictions for a synthetic load balancing environment for one of the following policies:


1. **Random**: a policy that assigns jobs to any server uniformly at random.

2. **Shortest Queue**:  Assign to the server with the smallest queue.

3. **Pow2**: Poll queue lengths of 2 servers (randomly) and assign to shortest queue 

4. **Pow3**: Poll queue lengths of 3 servers and assign to shortest queue.

5. **Pow4**: Poll queue lengths of 4 servers and assign to shortest queue.

6. **Pow5**: Poll queue lengths of 5 servers and assign to shortest queue.

7. **PseudoOptimal**: Normalize queue sizes with server rates and assign the job to the shortest normalized queue.

8. **Tracker**: Similar to PseudoOptimal, but estimates server rates with historical observations of processing times. 


To run the experiment and generate the counterfactuals of policy `POLICY_NAME` (while leaving the policy out of the training data), run:


```

python3 main.py --policy_out POLICY_NAME --dir CAUSALSIM_DIR --slsim_loss  mse_loss

```

Where `POLICY_NAME` can be any one of {
    "random",
    "Shortest queue",
    "pow2",
    "pow3",
    "pow4",
    "pow5",
    "PseudoOptimal",
    "Tracker"
}.

and slsim loss should be one of the three losses `mse_loss`, `l1_loss`, or `huber_loss`.

This will produce three plots (saved in `figures/`):

1. The MAPE of estimating the counterfactuals processing time of the jobs under the selected test policy using CausalSim and how it compares with SLSim.

2. The MAPE of estimating the counterfactual latencies of the jobs under the selected test policy using CausalSim and how it compares with SLSim.

3. CausalSim latent factors and how they compare with the actual job size. 


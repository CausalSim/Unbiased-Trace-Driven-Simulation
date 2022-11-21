# Real-world Adaptive BitRate (ABR) simulation with CausalSim

To reproduce the results in the paper, we need to:

0. Install required python packages
1. Download and prepare the dataset (~40GB).
2. Train CausalSim and SLSim models
3. Carry out counterfactual simulations with these models and tune hyper-parameters
4. Plot full-scale simulation results

***
## 0. Python Packages
We use Python (3.8 tested) for all experiments. Install the following packages via `pip` or `conda` :
```
numpy, pandas, tqdm, matplotlib, scikit-learn
```
Install PyTorch according to the website [instructions](https://pytorch.org).

---
## 1. Preparing the Dataset

First, create a directory as a workspace for datasets, models, simulations and plots. We'll call this directory CAUSALSIM_DIR.
Next, run the following command:
```
python3 data_preparation/create_dataset.py --dir CAUSALSIM_DIR
```
This script will download stream logs from [the puffer website](https://puffer.stanford.edu). It will then filter them
according to the [puffer-statistics](https://github.com/StanfordSNR/puffer-statistics) definition of `slow streams`.
The dataset is saved in `CAUSALSIM_DIR/cooked`.

To normalize the data and prepare it for training, run the following script:
```
python data_preparation/generate_subset_data.py --dir CAUSALSIM_DIR
```
---
## 2. Training
### Using the pre-trained checkpoints
We provide our trained checkpoints that we used in the paper in [assets](https://github.com/CausalSim/Unbiased-Trace-Driven-Simulation/tree/master/abr-puffer/assets).
To use them, copy everything inside [assets](https://github.com/CausalSim/Unbiased-Trace-Driven-Simulation/tree/master/abr-puffer/assets)
to `CAUSALSIM_DIR` and proceed to the next step (3. Counterfactual Simulation).
Unfortunately, we used random seeds for training these models, so training from scratch might produce different models.
To reproduce the exact results in the paper, use the pretrained checkpoints.

### Training from scratch
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]`, train the corresponding SLSim model with
the following script:
```
python training/sl_subset_train.py --dir CAUSALSIM_DIR --left_out_policy target 
```
Similarly, for each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` 
(loss hyper-parameter) from `[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, train the corresponding 
CausalSim model with the following script:
```
python training/train_subset.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
---
## 3. Counterfactual Simulation and Hyper-parameter Tuning
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` 
(loss hyper-parameter) from `[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, extract and save the 
latent factors using the corresponding CausalSim model with the following script:
```
python inference/extract_subset_latents.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa
```
To do counterfactual simulation with ExpertSim, use the following script:
```
python inference/expert_cfs.py --dir CAUSALSIM_DIR
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]`, use the corresponding trained SLSim model
for counterfactual simulation using the following script:
```
python inference/sl_subset_cfs.py --dir CAUSALSIM_DIR --left_out_policy target 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` (loss hyper-parameter) from 
`[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, use the corresponding CausalSim model to generate
counterfactual buffer and SSIM trajectories with the following script:
```
python inference/buffer_subset_cfs.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]`, calculate the average SSIM using the
ground-truth data, corresponding ExpertSim simulations, and corresponding SLSim simulations with the following scripts:
```
python analysis/original_subset_ssim.py --dir CAUSALSIM_DIR --left_out_policy target 
python analysis/sl_subset_ssim.py --dir CAUSALSIM_DIR --left_out_policy target 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` (loss hyper-parameter) from 
`[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, calculate the average SSIM using the corresponding
CausalSim simulations with the following script:
```
python analysis/subset_ssim.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` (loss hyper-parameter) from 
`[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, calculate the simulated buffer distribution's Earth 
Mover Distance (EMD) using the fround-truth data and corresponding CausalSim, SLSim, and ExpertSim simulations with the 
following script:
```
python analysis/subset_EMD.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
To Tune CausalSim's hyper-parameters for buffer and SSIM prediction, use the following script:
```
python analysis/tune_buffer_hyperparameters.py --dir CAUSALSIM_DIR  
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` (loss hyper-parameter) from 
`[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, use the corresponding CausalSim model to generate
counterfactual downloadtime trajectories with the following script:
```
python inference/downloadtime_subset_cfs.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]`, calculate the average stall ratio using 
the ground-truth data, corresponding ExpertSim simulations, and corresponding SLSim simulations with the following 
scripts:
```
python analysis/original_subset_stall.py --dir CAUSALSIM_DIR --left_out_policy target 
python analysis/sl_subset_stall.py --dir CAUSALSIM_DIR --left_out_policy target 
```
For each choice of `target` from `[linear_bba, bola_basic_v1, bola_basic_v2]` and `Kappa` (loss hyper-parameter) from 
`[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]`, calculate the average stall ratio using the 
corresponding CausalSim simulations with the following script:
```
python analysis/subset_stall.py --dir CAUSALSIM_DIR --left_out_policy target --C Kappa 
```
To Tune CausalSim's hyper-parameters for downloadtime prediction, use the following script:
```
python analysis/tune_downloadtime_hyperparameters.py --dir CAUSALSIM_DIR  
```
---
## 4. Result Visualization
You can find scripts for generating paper's plots in the [visualization folder](https://github.com/CausalSim/Unbiased-Trace-Driven-Simulation/tree/master/abr-puffer/visualization).
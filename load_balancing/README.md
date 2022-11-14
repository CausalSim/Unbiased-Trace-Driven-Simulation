# Load balancing experiment

## Run experiment

For the load balancing experiment, we generate counterfactual predictions for a synthetic load balancing environment for one of the following policies:

1. **Random**: a policy that assigns jobs to any server uniformly at random.
2. **Shortest Queue**:  Assign to the server with the smallest queue.
3. **Pow2**: Poll queue lengths of 2 servers (randomly) and assign to shortest queue 
4. **Pow3**: Poll queue lengths of 3 servers and assign to shortest queue.
5. **Pow4**: Poll queue lengths of 4 servers and assign to shortest queue.
6. **Pow5**: Poll queue lengths of 5 servers and assign to shortest queue.
7. **PseudoOptimal**: Normalize queue sizes with server rates and assign the job to the shortest normalized queue.
8. **Tracker**: Similar to PseudoOptimal, but estimates server rates with historical observations of processing times. 

To run the experiment and generate the counterfactuals of any of the aforementioned policies (while leaving the policy out of the training data), run:

```python
python3 main.py --policy_out {policy_name}
```
This will produce three plots (saved in `figures/`):
1. The MAPE of estimating the counterfactuals processing time of the jobs under the selected test policy using CausalSim and how it compares with SLSim.
2. The MAPE of estimating the counterfactual latencies of the jobs under the selected test policy using CausalSim and how it compares with SLSim.
3. CausalSim latent factors and how they compare with the actual job size. 


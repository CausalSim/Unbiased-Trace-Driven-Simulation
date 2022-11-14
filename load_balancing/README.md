# Load balancing experiment

## Run experiment

For the load balancing experiment, we generate counterfactual predictions for a synthetic load balancing environment for one of the following policies:

1. Random: a policy that assign jobs to any server uniformly at random.
2. Shortest Queue:  Assign to server with smallest queue.
3. Pow2: Poll queue lengths of 2 server (randomly) and assign to shortest queue 
4. Pow3: Poll queue lengths of 3 server and assign to shortest queue.
5. Pow4: Poll queue lengths of 4 server and assign to shortest queue.
6. Pow5: Poll queue lengths of 5 server and assign to shortest queue.
7. PseudoOptimal: Normalize queue sizes with server rates and assign to shortest normalized queue.
8. Tracker: Similar to PseudoOptimal, but estimates server rates with historical observations of processing times. 

To run the experiment and generate the counterfactuals of any of the aforementioned policies (while leaving the policy out of the training data), run:

```python
python3 main.py --policy_out {policy_name}
```
This will produce three plots (saved in `figure/`):
1. The MAPE of estimating the counterfactuals processing time of the jobs under the selected test policy using CausalSim and how it comapres with SLSim.
2. The MAPE of estimating the counterfactuals latencies of the jobs under the selected test policy using CausalSim and how it comapres with SLSim.
3. CausalSim latent factors and how they comapre with the actual job size. 


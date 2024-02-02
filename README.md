# Non-Stationary Latent AR Bandits
Code for "Non-Stationary Latent Autoregressive Bandits" paper.

Set experiment parameters such as `max_seed` or `num_time_steps` in the `global_params.py` file.

`ucb_agents.py` contains the code for all agents. It contains the parent class which implements standard UCB / OFUL for stochastic linear bandits as well as child classes such as standard UCB for multi-armed bandits and our algorithm presented in the paper.

`simulations.py` contains the skeleton code for running an experiment with an agent and an environment variant. There is also code for calculating the ground-truth optimal actions and mean rewards for each action.

To recreate the results in the main experiments section of our paper, run `python3 run_simulations.py`. We vary the true AR order k and the noise of the latent process. This script will save all results in the `experiment_results` folder.

To recreate the ablation results in section considering different mis-specified values of AR order k, run `run_sims_wrong_k.py`. This script will save all results in the `experiment_results` folder.

To generate the figures in our paper, run `python3 stats_and_figs/make_plots.py` and `python3 stats_and_figs/make_plots_wrong_k.py`. These scripts generate figures using results saved in `experiment_results` with correct k and mis-specified k, respectively.



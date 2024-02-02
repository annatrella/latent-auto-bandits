# Non-Stationary Latent AR Bandits
Code for "Non-Stationary Latent Autoregressive Bandits" paper.

Set experiment parameters such as `max_seed` or `num_time_steps` in the `global_params.py` file.

`ucb_agents.py` contains the code for all agents. It contains the parent class which implements standard UCB / OFUL for stochastic linear bandits as well as child classes such as standard UCB for multi-armed bandits and our algorithm presented in the paper.

`simulations.py` contains the skeleton code for running an experiment with an agent and an environment variant. There is also code for calculating the ground-truth optimal actions and mean rewards for each action.

`run_simulations.py` is the script for running for running experiments in the main results section of our paper. We vary the true AR order k and the noise of the latent process. This script will save all results in the `experiment_results` folder.

`run_sims_wrong_k.py` is the script for running simulations where we considered our algorithm with different mis-specified values of AR order k. This script will save all results in the `experiment_results` folder.

`stats_and_figs/make_plots.py` and `stats_and_figs/make_plots_wrong_k.py` creates the figures using experiment results saved in `experiment_results` with correct k and mis-specified k, respectively.



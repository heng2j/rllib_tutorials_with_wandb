# Let's get started with some basic imports.

import wandb

import ray  # .. of course
from ray import serve
from ray import tune
from ray.tune import Trainable
from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)

# Import the built-in RecSim exapmle environment: "Long Term Satisfaction", ready to be trained by RLlib.
from ray.rllib.examples.env.recommender_system_envs_with_recsim import LongTermSatisfactionRecSimEnv
from ray.rllib.agents.marwil import BCTrainer

from collections import OrderedDict
import gym  # RL environments and action/observation spaces
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from pprint import pprint
import re
import recsim  # google's RecSim package.
import requests
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import linregress, sem
from starlette.requests import Request
import tree  # dm_tree
print("imported all modules successfully")


# Offline RL training on LongTermSatisfactionRecSimEnv with wandb

api_key_file = "~/.wandb_api_key"

json_output_file = 'offline_rl/output-2022-03-28_16-43-43_worker-2_0.json' # TODO - use relative file path



offline_rl_env = LongTermSatisfactionRecSimEnv({
    "num_candidates": 20,
    "slate_size": 2,
    "wrap_for_bandits": False,  # SlateQ != Bandit
    "convert_to_discrete_action_space": False,
})



# Configuring the BCTrainer:
offline_rl_config = {
    # Specify your offline RL algo's historic (JSON) inputs:
    "input": [json_output_file],
    "actions_in_input_normalized": True,
    # Note: For non-offline RL algos, this is set to "sampler" by default.
    #"input": "sampler",

    # Since we don't have an environment and the obs/action-spaces are not defined in the JSON file,
    # we need to provide these here manually.
    "env": None,  # default
    "observation_space": offline_rl_env.observation_space,
    "action_space": offline_rl_env.action_space,

    # Perform "off-policy estimation" (OPE) on train batches and report results.
    "input_evaluation": ["is", "wis"],
}


def tune_function(api_key_file):
    """Example for using a WandbLoggerCallback with the function API"""
    analysis = tune.run(
        BCTrainer,
        metric="info/learner/default_policy/learner_stats/total_loss",
        mode="min",
        config=offline_rl_config,
        callbacks=[
            WandbLoggerCallback(api_key_file=api_key_file, project="Offline_RL_Wandb_example")
        ],
        stop={"training_iteration": 100}

    )
    return analysis.best_config



tune_function(api_key_file)
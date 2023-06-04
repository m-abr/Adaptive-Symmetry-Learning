from pathlib import Path
import os, yaml
from torch import nn
import numpy as np

class conf_cls():
    def __init__(self) -> None:
        pass

    def setup(self, params_rl_folder, read_env_only=False):

        if read_env_only:
            self.PARAMS_ENV_FILE = Path(params_rl_folder) / 'param_env.yaml'
        else:
            rl_path = Path(params_rl_folder)
            self.PARAMS_RL_FILE = rl_path / 'param_rl.yaml'
            self.PARAMS_ENV_FILE = rl_path.parent / 'param_env.yaml' # environment parameters should be in parent folder
            assert os.path.isfile(self.PARAMS_RL_FILE),  f"{self.PARAMS_RL_FILE} not found!"
        
        assert os.path.isfile(self.PARAMS_ENV_FILE), f"{self.PARAMS_ENV_FILE} not found!"

        #-------------------------------------------------------------- 1. Read ENV parameters
        with open(self.PARAMS_ENV_FILE) as f:
            self.ENV_params = yaml.load(f, Loader=yaml.loader.SafeLoader)

        if read_env_only:
            return
       
        #-------------------------------------------------------------- 2. Read RL parameters

        with open(self.PARAMS_RL_FILE) as f:
            params_PPO, params_sym_learning, params_sym_transformations = yaml.load(f, Loader=yaml.loader.SafeLoader)

        if params_sym_transformations is not None:
            for i, transformation in enumerate(params_sym_transformations):
                transformation["obs_indices"] = np.array(transformation["obs_indices"], dtype=int)              # Convert to numpy
                transformation["obs_multiplier"] = np.array(transformation["obs_multiplier"], dtype=np.float32) # Convert to numpy
                transformation["act_indices"] = np.array(transformation["act_indices"], dtype=int)              # Convert to numpy
                transformation["act_multiplier"] = np.array(transformation["act_multiplier"], dtype=np.float32) # Convert to numpy

                for param in transformation: # reuse parameters from 1st transformation
                    if isinstance(transformation[param], str) and transformation[param] == "reuse": # check if string to avoid warning
                        assert i != 0, "The first transformation cannot have reused parameters!"
                        transformation[param] = params_sym_transformations[0][param]

        self.RL_params = {
            'n_steps': params_PPO["batch_steps"],
            'clip_range': params_PPO["clip_range"],
            'ent_coef': params_PPO["entropy_coef"],
            'n_epochs': params_PPO["epochs"],
            'gae_lambda': params_PPO["lambda"],
            'gamma': params_PPO["gamma"],
            'learning_rate': params_PPO["learn_rate"],
            'max_grad_norm': params_PPO["max_grad_norm"],
            'batch_size': params_PPO["mini_batch_size"],
            'normalize_advantage': params_PPO["normalize_advantage"],
            'vf_coef': params_PPO["value_coef"],
            'policy_kwargs': {'log_std_init': params_PPO["NN_log_std_init"],
                            'ortho_init': params_PPO["NN_orthogonal_init"],
                            'activation_fn': getattr(nn, params_PPO["NN_activation"]),
                            'net_arch': [dict(pi=params_PPO["policy_network"], vf=params_PPO["value_network"])]},
            'sym_learning_args': {
                    'ga_form': params_sym_learning["ga_form"],                       # function form of g(a)
                    'HU_coef': params_sym_learning["HU_coef"],                       # cycle penalty function coefficient (maximum function value)   HU = HU_coef * HU_decay
                    'HU_decay': lambda x,f=params_sym_learning["HU_decay"]: eval(f), # cycle penalty function decay (decay due to cycle error)       HU = HU_coef * HU_decay
                    'HF': lambda x,f=params_sym_learning["HF"]: eval(f)              # action transformation penalty function
                } if params_sym_learning["use_sym_learning"] else None,
            'sym_transf_args': params_sym_transformations,
            }

        self.n_envs = params_PPO["environments"]
        self.total_time_steps = params_PPO["total_time_steps"]


Config = conf_cls()
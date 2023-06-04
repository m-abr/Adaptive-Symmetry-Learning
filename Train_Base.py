import os, time, math, csv, select, sys
from shutil import copy
import numpy as np
from datetime import datetime, timedelta
from typing import Callable
from itertools import count

import UI
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3 import PPO
import pickle
from config import Config


class Train_Base():
    def __init__(self) -> None:
        self.cf_last_time = 0
        self.cf_delay = 0
        self.cf_target_period = 0.0165 # as defined in gym_locomotion_envs

    def control_fps(self, read_input = False):
        ''' Add delay to control simulation speed '''

        if read_input:
            speed = input()
            if speed == '':
                self.cf_target_period = 0
                print(f"Changed simulation speed to MAX")
            else:
                if speed == '0':
                    inp = input("Paused. Set new speed or '' to use previous speed:")
                    if inp != '':
                        speed = inp   

                try:
                    speed = int(speed)
                    assert speed >= 0
                    self.cf_target_period = 0.02 * 100 / speed
                    print(f"Changed simulation speed to {speed}%")
                except:
                    print("""Train_Base.py: 
    Error: To control the simulation speed, enter a non-negative integer.
    To disable this control module, use test_model(..., enable_FPS_control=False) in your gym environment.""")

        now = time.time()
        period = now - self.cf_last_time
        self.cf_last_time = now
        self.cf_delay += (self.cf_target_period - period)*0.9
        if self.cf_delay > 0:
            time.sleep(self.cf_delay)
        else:
            self.cf_delay = 0


    def test_model(self, model:BaseAlgorithm, env, log_path:str=None, model_path:str=None, max_episodes=0, enable_FPS_control=True, verbose=1):
        '''
        Test model and log results

        Parameters
        ----------
        model : BaseAlgorithm
            Trained model 
        env : Env
            Gym-like environment
        log_path : str
            Folder where statistics file is saved, default is `None` (no file is saved)
        model_path : str
            Folder where it reads evaluations.npz to plot it and create evaluations.csv, default is `None` (no plot, no csv)
        max_episodes : int
            Run tests for this number of episodes
            Default is 0 (run until user aborts)
        verbose : int
            0 - no output (except if enable_FPS_control=True)
            1 - print episode statistics
        '''

        if model_path is not None:
            assert os.path.isdir(model_path), f"{model_path} is not a valid path"
            self.display_evaluations(model_path)

        if log_path is not None:
            assert os.path.isdir(log_path), f"{log_path} is not a valid path"

            # If file already exists, don't overwrite
            if os.path.isfile(log_path + "/test.csv"):
                for i in range(1000):
                    p = f"{log_path}/test_{i:03}.csv"
                    if not os.path.isfile(p):
                        log_path = p
                        break
            else:
                log_path += "/test.csv"
            
            with open(log_path, 'w') as f:
                f.write("reward,ep. length,rew. cumulative avg., ep. len. cumulative avg.\n")
            print("Train statistics are saved to:", log_path)

        if enable_FPS_control: # control simulation speed (using non blocking user input)
            print("\nThe simulation speed can be changed by sending a non-negative integer\n"
                  "(e.g. '50' sets speed to 50%, '0' pauses the simulation, '' sets speed to MAX)\n")

        ep_reward = 0
        ep_length = 0
        rewards_sum = 0
        reward_min = math.inf
        reward_max = -math.inf
        ep_lengths_sum = 0
        ep_no = 0

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            if enable_FPS_control: # control simulation speed (using non blocking user input)
                self.control_fps(select.select([sys.stdin], [], [], 0)[0]) 

            if done:
                obs = env.reset()
                rewards_sum += ep_reward
                ep_lengths_sum += ep_length
                reward_max = max(ep_reward, reward_max)
                reward_min = min(ep_reward, reward_min)
                ep_no += 1
                avg_ep_lengths = ep_lengths_sum/ep_no
                avg_rewards = rewards_sum/ep_no

                if verbose > 0:
                    print(  f"\rEpisode: {ep_no:<3}  Ep.Length: {ep_length:<4.0f}  Reward: {ep_reward:<6.2f}                                                             \n",
                        end=f"--AVERAGE--   Ep.Length: {avg_ep_lengths:<4.0f}  Reward: {avg_rewards:<6.2f}  (Min: {reward_min:<6.2f}  Max: {reward_max:<6.2f})", flush=True)
                
                if log_path is not None:
                    with open(log_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep_reward, ep_length, avg_rewards, avg_ep_lengths])
                
                if ep_no == max_episodes:
                    return

                ep_reward = 0
                ep_length = 0

    def learn_model(self, model:BaseAlgorithm, total_steps:int, path:str, eval_env=None, eval_freq=None, eval_eps=5, save_freq=None):
        '''
        Learn Model for a specific number of time steps

        Parameters
        ----------
        model : BaseAlgorithm
            Model to train
        total_steps : int
            The total number of samples (env steps) to train on
        path : str
            Path where the trained model is saved
            If the path already exists, an incrementing number suffix is added
        eval_env : Env
            Environment to periodically test the model
            Default is None (no periodical evaluation)
        eval_freq : int
            Evaluate the agent every X steps
            Default is None (no periodical evaluation)
        eval_eps : int
            Evaluate the agent for X episodes (both eval_env and eval_freq must be defined)
            Default is 5
        save_freq : int
            Saves model at every X steps
            Default is None (no periodical checkpoint)

        Returns
        -------
        model_path : str
            Directory where model was actually saved (considering incremental suffix)

        Notes
        -----
        If `eval_env` and `eval_freq` were specified:
            - The policy will be evaluated in `eval_env` every `eval_freq` steps
            - Evaluation results will be saved in `path` and shown at the end of training
            - Every time the results improve, the model is saved
        '''

        start = time.time()
        start_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # If path already exists, add suffix to avoid overwriting
        if os.path.isdir(path):
            for i in count():
                p = path.rstrip("/")+f'_{i:03}/'
                if not os.path.isdir(p):
                    path = p
                    break
        os.makedirs(path)

        # Backup parameter files
        copy(Config.PARAMS_RL_FILE,  os.path.join(path, os.path.basename(Config.PARAMS_RL_FILE )))
        copy(Config.PARAMS_ENV_FILE, os.path.join(path, os.path.basename(Config.PARAMS_ENV_FILE)))

        evaluate = bool(eval_env is not None and eval_freq is not None)

        # Create evaluation callback
        eval_callback = None if not evaluate else EvalCallback(eval_env, n_eval_episodes=eval_eps, eval_freq=eval_freq, log_path=path,
                                                               best_model_save_path=path, deterministic=True, render=False)

        # Create custom callback to display evaluations
        custom_callback = None if not evaluate else Cyclic_Callback(eval_freq, lambda:self.display_evaluations(path,True))

        # Create checkpoint callback
        checkpoint_callback = None if save_freq is None else CheckpointCallback(save_freq=save_freq, save_path=path, name_prefix="model", verbose=1)

        callbacks = CallbackList([c for c in [eval_callback, custom_callback, checkpoint_callback] if c is not None])

        model.learn( total_timesteps=total_steps, callback=callbacks )
        model.save( os.path.join(path, "last_model") )

        # Display evaluations if they exist
        if evaluate:
            self.display_evaluations(path)

        # Display timestamps + Model path
        end_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        duration = timedelta(seconds=int(time.time()-start))
        print(f"Train start:     {start_date}")
        print(f"Train end:       {end_date}")
        print(f"Train duration:  {duration}")
        print(f"Model path:      {path}")

        return path

    def display_evaluations(self, path, save_csv=False):

        eval_npz = os.path.join(path, "evaluations.npz")

        if not os.path.isfile(eval_npz):
            return

        console_width = 80
        console_height = 18
        symb_x = "\u2022"
        symb_o = "\u007c"
        symb_xo = "\u237f"

        with np.load(eval_npz) as data:
            time_steps = data["timesteps"]
            results_raw = np.mean(data["results"],axis=1)
            ep_lengths_raw = np.mean(data["ep_lengths"],axis=1)
        sample_no = len(results_raw)

        xvals = np.linspace(0, sample_no-1, 80)
        results    = np.interp(xvals, range(sample_no), results_raw)
        ep_lengths = np.interp(xvals, range(sample_no), ep_lengths_raw)

        results_limits    = np.min(results),    np.max(results)
        ep_lengths_limits = np.min(ep_lengths), np.max(ep_lengths)

        results_discrete    = np.digitize(results,    np.linspace(results_limits[0]-1e-5, results_limits[1]+1e-5,    console_height+1))-1
        ep_lengths_discrete = np.digitize(ep_lengths, np.linspace(0,                      ep_lengths_limits[1]+1e-5, console_height+1))-1

        matrix = np.zeros((console_height, console_width, 2), int)
        matrix[results_discrete[0]   ][0][0] = 1    # draw 1st column
        matrix[ep_lengths_discrete[0]][0][1] = 1    # draw 1st column
        rng = [[results_discrete[0], results_discrete[0]], [ep_lengths_discrete[0], ep_lengths_discrete[0]]]

        # Create continuous line for both plots
        for k in range(2):
            for i in range(1,console_width):
                x = [results_discrete, ep_lengths_discrete][k][i]
                if x > rng[k][1]:
                    rng[k] = [rng[k][1]+1, x]
                elif x < rng[k][0]:
                    rng[k] = [x, rng[k][0]-1]
                else:
                    rng[k] = [x,x]
                for j in range(rng[k][0],rng[k][1]+1):
                    matrix[j][i][k] = 1

        print(f'{"-"*console_width}')
        for l in reversed(range(console_height)):
            for c in range(console_width):
                if   np.all(matrix[l][c] == 0): print(end=" ")
                elif np.all(matrix[l][c] == 1): print(end=symb_xo)
                elif matrix[l][c][0] == 1:      print(end=symb_x)
                else:                           print(end=symb_o)
            print()
        print(f'{"-"*console_width}')
        print(f"({symb_x})-reward          min:{results_limits[0]:11.2f}    max:{results_limits[1]:11.2f}")
        print(f"({symb_o})-ep. length      min:{ep_lengths_limits[0]:11.0f}    max:{ep_lengths_limits[1]:11.0f}    {time_steps[-1]/1000:15.0f}k steps")
        print(f'{"-"*console_width}')

        # save CSV
        if save_csv:
            eval_csv = os.path.join(path, "evaluations.csv")
            with open(eval_csv, 'a+') as f:
                writer = csv.writer(f)
                if sample_no == 1:
                    writer.writerow(["time_steps", "reward ep.", "length"])
                writer.writerow([time_steps[-1],results_raw[-1],ep_lengths_raw[-1]])


    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        '''
        Linear learning rate schedule

        Parameters
        ----------
        initial_value : float
            Initial learning rate
        
        Returns
        -------
        schedule : Callable[[float], float]
            schedule that computes current learning rate depending on remaining progress
        '''
        def func(progress_remaining: float) -> float:
            '''
            Compute learning rate according to current progress

            Parameters
            ----------
            progress_remaining : float
                Progress will decrease from 1 (beginning) to 0
            
            Returns
            -------
            learning_rate : float
                Learning rate according to current progress
            '''
            return progress_remaining * initial_value

        return func

    @staticmethod
    def export_model(input_file, output_file, add_sufix=True):
        '''
        Export model weights to binary file

        Parameters
        ----------
        input_file : str
            Input file, compatible with algorithm
        output_file : str
            Output file, including directory
        add_sufix : bool
            If true, a suffix is appended to the file name: output_file + "_{index}.pkl"
        '''

        # If file already exists, don't overwrite
        if add_sufix:
            for i in count():
                f = f"{output_file}_{i:03}.pkl"
                if not os.path.isfile(f):
                    output_file = f
                    break
        
        model = PPO.load(input_file)
        weights = model.policy.state_dict() # dictionary containing network layers

        w = lambda name : weights[name].detach().cpu().numpy() # extract weights from policy

        var_list = []
        for i in count(0,2): # add hidden layers (step=2 because that's how SB3 works)
            if f"mlp_extractor.policy_net.{i}.bias" not in weights:
                break
            var_list.append([w(f"mlp_extractor.policy_net.{i}.bias"), w(f"mlp_extractor.policy_net.{i}.weight"), "tanh"])

        var_list.append( [w("action_net.bias"), w("action_net.weight"), "none"] ) # add final layer
        
        with open(output_file,"wb") as f:
            pickle.dump(var_list, f, protocol=4) # protocol 4 is backward compatible with Python 3.4



class Cyclic_Callback(BaseCallback):
    ''' Stable baselines custom callback '''
    def __init__(self, freq, function):
        super(Cyclic_Callback, self).__init__(1)
        self.freq = freq
        self.function = function

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.function()
        return True # If the callback returns False, training is aborted early

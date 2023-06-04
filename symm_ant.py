from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from Train_Base import Train_Base
import os, gym, math, sys
import numpy as np
import pybullet_envs
import shutil
from config import Config

class GymWrapper():
    # gym used for training 

    def __init__(self, name, instance_no, evaluate=False, render=False) -> None:
        self.name = name
        self.env = gym.make(name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.terminate_if_no_progress = Config.ENV_params["terminate_if_no_progress"]

        if render:
            self.env.render(mode="human")

        # biased action
        self.action_mult = np.array(Config.ENV_params["action_multiplier"], np.float32)
        
        # divide action space limits [-1,1] by the action multiplier if adapt_action_space is True
        if Config.ENV_params["adapt_action_space"]:
            self.action_space = gym.spaces.Box(low=-1/self.action_mult, high=1/self.action_mult, dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        self.metadata = self.env.metadata
        self.evaluate = evaluate
        self.instance_no = instance_no

        target_angles = [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, -np.pi*3/4, -np.pi/2, -np.pi/4]
        self.train_targets = [target_angles[i] for i in Config.ENV_params["train_targets"]]  # must be a subset of Config.ENV_params["available_targets"]

        # modify the mass of the robot's feet
        unwrapped = self.env.unwrapped
        unwrapped.reset()
        self.p = p = unwrapped._p

        self.robot_torso = part = unwrapped.robot.parts['torso']
        bodyIndex = part.bodyIndex
        FOOT_INDICES = [4,9,14,19]
        KNEE_INDICES = [3,8,13,18]

        for i in Config.ENV_params["feet_mass"]:
            p.changeDynamics(part.bodies[bodyIndex], FOOT_INDICES[i[0]], mass=i[1])


        


    def reset(self):
        while True:
            obs = self.env.reset()
            self.init_target_vec = obs[1:3].copy()
            init_ori = np.arctan2(*self.init_target_vec) # initial orienation in radians
            if self.evaluate or any(np.pi - abs( abs(t-init_ori)-np.pi ) < 0.05 for t in self.train_targets): # restrict train targets
                break
        self.best_x = -1
        self.watchdog = 0
        return obs

    def step(self,action):
        new_ac = action * self.action_mult
        obs, reward, done, info = self.env.step(new_ac)
        deviation = np.arccos(np.clip(np.dot(obs[1:3], self.init_target_vec),-1,1)) * 180 / math.pi # deviation from initial angle in degrees
        if deviation > 25:
            done = True

        if self.terminate_if_no_progress:
            pos_x = self.p.getBasePositionAndOrientation(self.robot_torso.bodies[self.robot_torso.bodyIndex])[0][0]
            if pos_x < self.best_x+0.01:
                self.watchdog += 1
                if self.watchdog > 30:
                    done = True
            else:
                self.watchdog = 0
                self.best_x = pos_x
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class Train(Train_Base):
    def __init__(self) -> None:
        super().__init__()


    def train(self, model_name, instance_no):

        #--------------------------------------- Learning parameters

        gym_name = "AntBulletEnv-v0"
        instance_no = int(instance_no)
        k_t_factor = 10 # time steps for neutral state rejection (x batch steps)

        folder_name = f'{gym_name}_{model_name}_inst{instance_no}'
        model_path = f'./models/{folder_name}/'

        if instance_no == 0: 
            print("Model path:", model_path)

        #--------------------------------------- Create Env(s)
        def init_env():
            def thunk():
                return GymWrapper(gym_name, instance_no)
            return thunk

        if Config.n_envs == 1:
            env = GymWrapper(gym_name, instance_no)
            eval_env = GymWrapper(gym_name, -1, True)
        else:
            env = SubprocVecEnv( [init_env() for _ in range(Config.n_envs)] )
            eval_env = SubprocVecEnv( [init_env()] )

        #--------------------------------------- Run algorithm

        batch_dir = f'./models/{model_name}'
        if not os.path.isdir(batch_dir):
            os.makedirs(batch_dir, exist_ok=True)

        s_log = f'{batch_dir}/sym_evaluations_{instance_no}.csv'

        model = PPO( policy='MlpPolicy', env=env, k_t_factor=k_t_factor, verbose=1, device='cpu', 
            symmetry_log=s_log, symmetry_log_freq=5, **Config.RL_params)

        model_path = self.learn_model( model, Config.total_time_steps, model_path, eval_env=eval_env, 
                                      eval_freq=Config.RL_params['n_steps']*15, eval_eps=16  )
  
        # copy evaluations to batch directory
        shutil.copyfile(os.path.join(model_path, "evaluations.csv"), f'{batch_dir}/evaluations_{instance_no}.csv')
        
        env.close()
        eval_env.close()
        

    def test(self, file):

        gym_name = "AntBulletEnv-v0"

        env = GymWrapper(gym_name, 0, True, render=True)
        env.render()
        model = PPO.load( file, env=env )

        folder = os.path.dirname(file)
        self.test_model( model, env, log_path=folder, model_path=folder )

        env.close()



if __name__ == "__main__":
    if not 2 < len(sys.argv) < 5: # must be 2 or 3 args in addition to script path
        print("Usage:\nTrain:\t python symm_ant.py model_name params_folder instance_number")
        print("Test:\t python symm_ant.py model_file params_folder")
    else:
        t = Train()
        model = sys.argv[1]
        testing = (len(sys.argv) == 3)
        Config.setup(sys.argv[2], testing)

        if testing:
            print("Testing model:", model)
            t.test(model)
        else:
            instance_no = sys.argv[3]
            print("Training new model:", model, "instance:", instance_no)

            t.train(model, instance_no)
            print("Batch folder name where to save all evaluations:", instance_no)


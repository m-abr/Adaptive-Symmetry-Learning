## Implementation of symmetry extensions for PPO
- **Adaptive Symmetry Learning (ASL)** [1]
- **Proximal Symmetry Loss (PSL)** [2]
- **Mirror Symmetry Loss (MSL)** [3]


ASL is a novel symmetry extension for actor-critic methods that directs the policy to perform symmetric actions in symmetric states. The advantage in relation to similar extensions is the ability to learn and compensate asymmetries in the robot, environment, reward function or agent (control process). Another key strength is its performance when generalizing knowledge to hidden symmetric states. (see [1] for more details)

[1] Abreu, M., Reis, L. P., Lau, N. (2023). Symmetry-learning actor-critic extension to handle imperfect symmetry: case study of an ant robot. arXiv preprint \
[2] Kasaei, M., Abreu, M., Lau, N., Pereira, A., & Reis, L. P. (2021). A CPG-based agile and versatile locomotion framework using proximal symmetry loss. arXiv preprint arXiv:2103.00928. \
[3] Yu, W., Turk, G., & Liu, C. K. (2018). Learning symmetric and low-energy locomotion. ACM Transactions on Graphics (TOG), 37(4), 1-12.

## Implementation details

Codebase (included in this project):
- Stable Baselines3 (SB3) v1.6.1a4 
    - Fork of [this commit](https://github.com/DLR-RM/stable-baselines3/commit/fdca786f0999ea402ad5da98e3d69aa68bcbf635) (Sep 2, 2022)
- PyBullet Envs from Pybullet v3.2.5
    - Copy of bullet3/examples/pybullet/gym/pybullet_envs/ from [this release](https://github.com/bulletphysics/bullet3/releases/tag/3.25)

Modifications to imported code:
- Stable Baselines3 (SB3) v1.6.1a4 
    - /stable_baselines3/ppo/sym.py (created) — Implementation of symmetry extensions
    - /stable_baselines3/ppo/ppo.py (modified) — Calls sym.py, adds symmetry loss function to PPO's loss function
    - /stable_baselines3/common/buffers.py (modified) — small change to return indices of next mini-batch
- PyBullet Envs v3.2.5
    - /pybullet_envs/gym_locomotion_envs.py (modified) — generates several scenarios based on the original AntBulletEnv-v0 environment
- Mirror Symmetry Loss (MSL) 
    - added value loss to symmetry loss function
    - added support for involutory transformations
- Proximal Symmetry Loss (PSL)
    - added value loss to symmetry loss function


## Requirements

```
sudo apt install python3 python3-pip
pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install gym==0.21.0 pandas matplotlib autograd scipy pyyaml pybullet
```


Tested on a clean Ubuntu 20.04 installation with:
- Python 3.8.10
- PyTorch 1.13.1
- Gym 0.21.0  
- Pandas 2.0.1
- Matplotlib 3.7.1
- Autograd 1.5
- Scipy 1.10.1
- PyYAML 6.0
- PyBullet 3.2.5


## Recommended

To reduce number of parallel numpy threads:
```
export OMP_NUM_THREADS=1
```

## Examples using ant gym
<br />

#### Train a single model in scenario **A1.1** using ASL
**usage:**  symm_ant.py new_model_name params_folder instance_no
```
python3 symm_ant.py my_model parameters/A1.1/ASL 0
```

- The previous command will read the scenario specification from **/parameters/A1.1/param_env.yaml** and the learning hyperparameters from **/parameters/A1.1/ASL/param_rl.yaml**
- Check **/parameters/** folder for different scenarios. Each scenario has tuned parameters for 4 algorithms: vanilla PPO ("**None**"), PPO+ASL ("**ASL**"), PPO+MSL ("**MSL**"), and PPO+PSL ("**PSL**").

<br />

#### Load model (see it in action + statistics in console)

- By default the policy is evaluated every 15 iterations, saving the best model so far as "best_model.zip". The final model is called "last_model.zip".
- The simulation runs in real-time. To change the speed, write a number in the console betweem 0-1000 and press ENTER, where 0 means stopped, 10 means 10% speed, 100 is real-time, 101 is 101% speed and so on. 
- Pressing ENTER without entering a number means "run as fast as possible"

```
python3 symm_ant.py models/AntBulletEnv-v0_my_model_inst0/best_model.zip ./parameters/A1.1
```

<br />

#### Train a batch of independent models with same configuration

This command trains 12 models with vanilla PPO on scenario A6.1:

```
./ant_batch.sh my_batch parameters/A6.1/None
```

This example will generate a folder per model plus a common folder where all statistics are gathered at the end of training:


> models/AntBulletEnv-v0_my_batch_inst0/ \
> models/AntBulletEnv-v0_my_batch_inst1/ \
> (...) \
> models/AntBulletEnv-v0_my_batch_inst11/ \
> models/my_batch/

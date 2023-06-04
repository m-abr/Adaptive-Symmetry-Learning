from collections import deque
import torch as th
import csv
import numpy as np
import autograd.numpy as gnp
from autograd import jacobian, hessian
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import math
from torch.nn import functional as F


class Single():
    def __init__(self) -> None:
        self.sym_idx_list = []
        self.target = None
        self.targ_hist = deque(maxlen=5)

    def get_target_spread(self, target):
        self.targ_hist.append(target)
        std = np.std(self.targ_hist, axis=0)

        # absolute value of coef. of variation
        return np.sum(std / (np.abs(np.mean(self.targ_hist, axis=0))+0.1))

class Pair():

    def __init__(self, ID) -> None:
        self.ID = ID
        self.sym_idx_01_dir = []
        self.sym_idx_01_inv = []
        self.sym_idx_10_dir = []
        self.sym_idx_10_inv = []
        self.inp = None
        self.out = None
        self.raw_target = None
        self.heterogeneity = 0
        self.target_m = 0
        self.target_b = 0
        self.cycle_decay = 0
        self.func_weight = 0
        self.targ_hist = deque(maxlen=5)

    def get_target_spread(self, target):
        self.targ_hist.append(target)
        std = np.std(self.targ_hist, axis=0)

        # absolute value of coef. of variation
        return np.sum(std / (np.abs(np.mean(self.targ_hist, axis=0))+0.1))

    
    def add_sym_plane(self, is_01, is_direct, sym_idx):
        # multiplier index is implicit in self.ID
        if is_01: 
            if is_direct:
                self.sym_idx_01_dir.append(sym_idx)
            else:
                self.sym_idx_01_inv.append(sym_idx)
        else:
            if is_direct:
                self.sym_idx_10_dir.append(sym_idx)
            else:
                self.sym_idx_10_inv.append(sym_idx)


class Sym():
    def __init__(self, ppo, sym_transf_args, sym_learning_args, symmetry_log, symmetry_log_freq, k_t_factor) -> None:
        np.set_printoptions(precision=2, suppress=True)
        self.symmetry_log = None # when sym_transf_args is None, this is the default value
        self.symmetry_log_freq = symmetry_log_freq
        self.ac_dim = ppo.env.action_space.shape[0] # action length
        self.ppo = ppo
        self.sym_transf_args = sym_transf_args

        self.is_sym_learn_enabled = False
        if sym_learning_args is not None:
            self.is_sym_learn_enabled = True
            self.ga_form = sym_learning_args["ga_form"]
            assert self.ga_form in ["mx+b","mx"]
            self.HU_coef = sym_learning_args["HU_coef"]
            self.HU_decay = sym_learning_args["HU_decay"] 
            self.HF = sym_learning_args["HF"] # action transformation penalty function (evaluates target stability)

        graph = {i:set() for i in range(self.ac_dim)} # graph of action relations (to find cycles)
        
        if sym_transf_args is not None:
            self.pairs : list[Pair] = []
            self.singles : dict[int,Single] = dict()
            self.pairs_to_index = dict() # convert pair_key to index
            self.ob_history = deque(maxlen=k_t_factor)
            for sk_idx, sk in enumerate(sym_transf_args):
                sk["act_indices"] = th.from_numpy(sk["act_indices"]).to(ppo.device)
                sk["obs_indices"] = th.from_numpy(sk["obs_indices"]).to(ppo.device)
                sk["act_multiplier"] = th.from_numpy(sk["act_multiplier"]).to(ppo.device)
                sk["obs_multiplier"] = th.from_numpy(sk["obs_multiplier"]).to(ppo.device)
                sk["func_weight"] = th.ones(self.ac_dim, dtype=th.float32, device=ppo.device)
                sk["act_bias"] = th.zeros(self.ac_dim, dtype=th.float32, device=ppo.device)

                # Prepare auxiliary variables for ASL
                if sk["algorithm"] == "asl":
                    sk["k_v_alpha"] = (sk["k_v_gate_coef"]*sk["k_v_gate_coef"]+1) / (2*sk["k_v_gate_coef"]) # auxiliary constant (value gate)
                    sk["k_v_minus_alpha"] = sk["k_v_gate_coef"] - sk["k_v_alpha"] # auxiliary constant (value gate)

                for a in range(self.ac_dim):
                    b = sk["act_indices"][a].item()
                    mult = sk["act_multiplier"][a] # a = b * mult  (the bias doesn't change this relation)
                    
                    # if self relations have bias, they are not involutory, and there is no way of knowing the direction
                    if self.is_sym_learn_enabled and a == b and self.ga_form == "mx+b": 
                        if mult == -1:
                            if a not in self.singles:
                                self.singles[a] = Single()
                            self.singles[a].sym_idx_list.append(sk_idx) # all appended relations are a->a, m=-1 (inverse)
                        continue

                    if a == b or mult == 0: # no self relations for "mx", mult == 0 does not define a finite relation
                        continue 
                    
                    pair_ID = (a,b) if a < b else (b,a)
                    if pair_ID not in self.pairs_to_index:
                        self.pairs_to_index[pair_ID] = len(self.pairs_to_index)
                        self.pairs.append( Pair(pair_ID) ) 
                        self.pairs[-1].add_sym_plane(a > b, mult > 0, sk_idx) # b->a always, so (a,b) means 1->0, (b,a) means 0->1
                    else:
                        self.pairs[self.pairs_to_index[pair_ID]].add_sym_plane(a > b, mult > 0, sk_idx)

            self.pair_no = len(self.pairs)
            self.iteration_counter = 0

            # Find cycles
            for p in self.pairs:
                graph[p.ID[0]].add(p.ID[1])
                graph[p.ID[1]].add(p.ID[0])
            self.groups_cycles = self.find_cycles(graph)

            self.symmetry_log = symmetry_log
            self.sym_log_countdown = 0
            if symmetry_log is not None:
                with open(self.symmetry_log, 'w') as f:
                    writer = csv.writer(f)
                    cols = ["time_steps", "std_dev"]
                    cols.extend([f"{i}_sy_val_dif" for i in range(len(sym_transf_args))])
                    cols.extend([f"{i}_dead_ratio" for i in range(len(sym_transf_args))])
                    cols.extend([f"{i}_value_weight" for i in range(len(sym_transf_args))])
                    cols.extend([f"single_targ_b_{p}" for p in self.singles])
                    cols.extend([f"pair_targ_m_{p.ID[0]}{p.ID[1]}" for p in self.pairs])
                    cols.extend([f"pair_targ_b_{p.ID[0]}{p.ID[1]}" for p in self.pairs])
                    cols.extend([f"pair_cycdec_{p.ID[0]}{p.ID[1]}" for p in self.pairs])
                    cols.extend([f"pair_heterog_{p.ID[0]}{p.ID[1]}" for p in self.pairs])
                    cols.extend([f"pair_func_w_{p.ID[0]}{p.ID[1]}" for p in self.pairs])
                    cols.extend([f"{i}_mult_{j}" for i in range(len(sym_transf_args)) for j in range(self.ac_dim)])
                    cols.extend([f"{i}_bias_{j}" for i in range(len(sym_transf_args)) for j in range(self.ac_dim)])
                    writer.writerow(cols)

    def find_cycles(self,graph):
        def _canonical(path):
            # returns canonical path for comparison
            # starts from smallest item in the direction of smaller item
            index_min = min(range(len(path)), key=path.__getitem__)
            next = index_min+1 if index_min+1 < len(path) else 0
            prev = index_min-1 if index_min > 0 else len(path)-1

            if path[next] < path[prev]: # note: they cannot be equal
                return tuple(path[index_min:] + path[:index_min])
            else: # move in the opposite direction
                aux = path[index_min+1:] + path[:index_min+1]
                aux.reverse()
                return tuple(aux)

        paths = set()  # reduced set of canonical paths

        for v in graph:
            stack = [(v, [])]
            while stack:
                state, path = stack.pop()
                if path and state == v:
                    if len(path) > 2:
                        paths.add( _canonical(path) )  
                    continue
                for next_state in graph[state]:
                    if next_state in path:
                        continue
                    stack.append((next_state, path+[next_state]))

        
        pair_paths = []
        ptoi = lambda i: self.pairs_to_index[i] # pair to index
        for p in paths:
            pair_path = ([],[]) # pair_idx (0->1), pair_idx (1->0)
            for i in range(len(p)-1):
                src = 0 if p[i] < p[i+1] else 1
                pair_path[src].append( ptoi((p[i+1],p[i])) if src else ptoi((p[i],p[i+1])) ) # pair_idx
            src = 0 if p[-1] < p[0] else 1
            pair_path[src].append( ptoi((p[0],p[-1])) if src else ptoi((p[-1],p[0])) )       # last pair_idx
            pair_paths.append(pair_path)

        return pair_paths

    def before_optimization(self, clip_range):

        ppo = self.ppo
        self.iteration_counter += 1
        batch_obs_np = ppo.rollout_buffer.swap_and_flatten(ppo.rollout_buffer.observations).copy()
        batch_obs = th.from_numpy(batch_obs_np).to(ppo.device)
        batch_values = ppo.rollout_buffer.swap_and_flatten(ppo.rollout_buffer.values).flatten() # already numpy
        self.batch_mu_old_of_obs = ppo.policy(batch_obs, deterministic=True)[0].detach().clone() # mean action
        batch_acs = th.from_numpy(ppo.rollout_buffer.swap_and_flatten(ppo.rollout_buffer.actions).copy()).to(ppo.device)
        batch_adv = th.from_numpy(ppo.rollout_buffer.swap_and_flatten(ppo.rollout_buffer.advantages).flatten().copy()).to(ppo.device)

        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if ppo.normalize_advantage and len(batch_adv) > 1:
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
        
        if self.sym_transf_args[0]["algorithm"] == "asl":
            
            stdev_old = th.exp( ppo.policy.log_std.detach().clone() ) # stdev right after collecting a batch (no grad)
            self.ob_history.append(batch_obs_np)
            ob_hist_mean = np.mean(self.ob_history, axis=(0,1)) # mean per dimension
            ob_hist_mad = th.from_numpy(np.mean(np.abs((ob_hist_mean-self.ob_history)),axis=(0,1))).to(ppo.device) # mean absolute deviation

        
        for sk in self.sym_transf_args:
            with th.no_grad():

                sobs = sk["sym_obs"] = th.multiply( th.index_select(batch_obs, 1, sk["obs_indices"]), sk["obs_multiplier"])
                sk["mu_old_of_sym_obs"] = ppo.policy(sobs, deterministic=True)[0].detach().clone() # mean actions for sym obs
                sk["swapped_mu_old_of_obs"] = th.index_select(self.batch_mu_old_of_obs, 1, sk["act_indices"])
                sk["sym_mu_old_of_obs"] = th.multiply( sk["swapped_mu_old_of_obs"], sk["act_multiplier"]) + sk["act_bias"] # sym of mean actions
              
                if sk["algorithm"] == "asl":
                    sk["dead_zone_gate"] = th.ones(batch_obs.shape[0], dtype=th.float32, device=ppo.device)
                    k_dev = sk["k_dead_zone"]

                    if sk["dead_zone_mode"] == 1: # only neutral states
                        sk["dead_zone_gate"] *= ( th.mean(th.abs(batch_obs - sobs)/ob_hist_mad,dim=1) > k_dev ).float()
                    elif sk["dead_zone_mode"] == 2: # rem sym obs if it was experienced (includes neutral states)
                        for bob in batch_obs: # remove all sym obs that are similar an original ob, including self similarity (similar means near in every dimension)    
                            sk["dead_zone_gate"] *= ( th.mean(th.abs(bob - sobs)/ob_hist_mad,dim=1) > k_dev ).float()

                        # this alternative is much faster but uses a prohibitive amount of memory
                        #aux = th.any( th.abs(th.unsqueeze(sobs, 1) - batch_obs) > k_dev * ob_hist_mad, dim=2).float() # unsqueeze adds dimension to sobs to broadcast operation
                        #sk["dead_zone_gate"] = th.prod(aux, 0)

                    sk["dead_ratio"] =  1 - th.sum(sk["dead_zone_gate"]).item() / batch_obs.shape[0]

                    # compute delta mu
                    E = math.pow(1+clip_range,1/self.ac_dim) - 1
                    sk["delta_mu"] = sk["k_shift"] * stdev_old * math.sqrt( -2*math.log( 1/(1+E) ) )

                else:
                    sk["dead_ratio"] = 0

                # PSL: pi_old( g(am) | f(s) )                       ---> but also relevant for symmetry evaluation
                sk["pi_old_of_sym_obs_and_sym_mu_old_of_obs"] = ppo.policy.get_distribution(sobs).log_prob(sk["sym_mu_old_of_obs"]).detach().clone() 

                if self.sym_log_countdown == 0: # Evaluate symmetry (l1 and l2), considering rotations
                    v_of_sym_obs = ppo.policy.predict_values(sobs).detach().cpu().numpy().flatten() # used for value loss
                    sk["v_dist"] = np.mean(np.abs(batch_values - v_of_sym_obs))  # for time steps we get the abs mean

        if self.is_sym_learn_enabled:    
            self.learn_symmetry()
        

    def learn_symmetry(self):
        print("Learning symmetry")

        if self.ga_form == "mx+b":
            for i, single in self.singles.items():

                inp, out = [],[]

                for sk_idx in single.sym_idx_list:
                    inp.append( self.batch_mu_old_of_obs[:,i] )
                    out.append( self.sym_transf_args[sk_idx]["mu_old_of_sym_obs"][:,i] )

                # combine singles sources
                inp = th.cat(inp).numpy()
                out = th.cat(out).numpy()

                single.target = (np.sum(inp, axis=0) + np.sum(out, axis=0)) / len(inp) # value of b
                

        for i, pair in enumerate(self.pairs):
            
            # Default Transformation: 0->1 direct 
            inp, out = [],[]

            for sk_idx in pair.sym_idx_01_dir: # 0->1 direct
                inp.append( self.sym_transf_args[sk_idx]["swapped_mu_old_of_obs"][:,pair.ID[1]] )
                out.append( self.sym_transf_args[sk_idx]["mu_old_of_sym_obs"][:,pair.ID[1]] )

            for sk_idx in pair.sym_idx_01_inv: # 0->1 inverse
                inp.append( -self.sym_transf_args[sk_idx]["swapped_mu_old_of_obs"][:,pair.ID[1]] ) # invert input
                out.append( self.sym_transf_args[sk_idx]["mu_old_of_sym_obs"][:,pair.ID[1]] ) 

            for sk_idx in pair.sym_idx_10_dir: # 1->0 direct
                inp.append( self.sym_transf_args[sk_idx]["mu_old_of_sym_obs"][:,pair.ID[0]] )
                out.append( self.sym_transf_args[sk_idx]["swapped_mu_old_of_obs"][:,pair.ID[0]] )

            for sk_idx in pair.sym_idx_10_inv: # 1->0 inverse
                inp.append( -self.sym_transf_args[sk_idx]["mu_old_of_sym_obs"][:,pair.ID[0]] ) # invert input
                out.append( self.sym_transf_args[sk_idx]["swapped_mu_old_of_obs"][:,pair.ID[0]] )

            # combine pair sources
            pair.inp = inp = th.cat(inp).numpy()
            pair.out = out = th.cat(out).numpy()

            if self.ga_form == "mx+b":
                sum_io = np.sum(inp*out, axis=0)
                sum_ii = np.sum(inp*inp, axis=0)
                sum_i = np.sum(inp, axis=0)
                sum_o = np.sum(out, axis=0)
                n = len(inp)
                m = (n*sum_io - sum_i*sum_o) / (n*sum_ii - sum_i*sum_i)
                b = (sum_o - m*sum_i) / n
                pair.raw_target = (m,b)
            elif self.ga_form == "mx":
                pair.raw_target = np.sum(inp*out, axis=0)/(np.sum(inp*inp, axis=0)+1e-8)

        for i in range(self.pair_no):
            self.pairs[i].cycle_decay = 1   

        if self.ga_form == "mx+b":
            targets = []
            for p in self.pairs:
                targets.extend(p.raw_target)
            targets = np.array(targets, float)

            for pair_list, pair_inv_list in self.groups_cycles:
                dev = 1
                for i in pair_list:
                    m,b = targets[i*2:i*2+2]
                    dev = dev * m + b
                for i in pair_inv_list:    
                    m,b = targets[i*2:i*2+2]
                    dev = (dev - b) / (m+1e-8)
                dev = np.abs(dev - 1)

                cd = self.HU_decay(dev)
                
                for i in pair_list+pair_inv_list:
                    self.pairs[i].cycle_decay = min(self.pairs[i].cycle_decay, cd) # penalize pairs in cycle

            # Optimization: force values to obey cycles (use previous chunk decision)
            if False:
                targets = self.OPT_mxb(targets)
                    
            # update action multipliers
            for i, pair in enumerate(self.pairs):
                s = self.sym_transf_args
                m = targets[i*2]
                b = targets[i*2+1]

                # importance of pair
                diff = pair.get_target_spread(np.copy(targets[i*2:i*2+2]))
                pair.func_weight = self.HF(diff)

                pair.target_m = m # for stats
                pair.target_b = b # for stats


                mean_abs_out = np.mean(np.abs(pair.out)) # free variable
                pair.heterogeneity = np.mean(np.abs(pair.inp*m+b-pair.out)) / mean_abs_out

                if m == 0: # If the target is zero, it's a bad target 
                    continue 

                update_w = self.HU_coef * pair.cycle_decay
                dim = pair.ID[1]
                for sk_idx in pair.sym_idx_01_dir: # 0->1 (positive correlation)
                    
                    s[sk_idx]["act_multiplier"][dim] += update_w * (m - s[sk_idx]["act_multiplier"][dim])
                    s[sk_idx]["func_weight"][dim] = pair.func_weight
                    s[sk_idx]["act_bias"][dim] += update_w * (b - s[sk_idx]["act_bias"][dim])

                for sk_idx in pair.sym_idx_01_inv: # 0->1 reflection about Y-axis
                    s[sk_idx]["act_multiplier"][dim] += update_w * (-m - s[sk_idx]["act_multiplier"][dim])
                    s[sk_idx]["func_weight"][dim] = pair.func_weight
                    s[sk_idx]["act_bias"][dim] += update_w * (b - s[sk_idx]["act_bias"][dim])

                dim = pair.ID[0]
                for sk_idx in pair.sym_idx_10_dir: # 1->0 (positive correlation)
                    s[sk_idx]["act_multiplier"][dim] += update_w * (1/m - s[sk_idx]["act_multiplier"][dim])
                    s[sk_idx]["func_weight"][dim] = pair.func_weight
                    s[sk_idx]["act_bias"][dim] += update_w * (-b/m - s[sk_idx]["act_bias"][dim])

                for sk_idx in pair.sym_idx_10_inv: # 1->0 reflection about Y-axis
                    s[sk_idx]["act_multiplier"][dim] += update_w * (-1/m - s[sk_idx]["act_multiplier"][dim])
                    s[sk_idx]["func_weight"][dim] = pair.func_weight
                    s[sk_idx]["act_bias"][dim] += update_w * (b/m - s[sk_idx]["act_bias"][dim])

            # update singles bias
            for i, single in self.singles.items():
                s = self.sym_transf_args
                b = single.target

                # importance of pair
                diff = single.get_target_spread(float(b))
                func_weight = self.HF(diff)

                for sk_idx in single.sym_idx_list:
                    s[sk_idx]["func_weight"][i] = func_weight
                    s[sk_idx]["act_bias"][i] += self.HU_coef * (b - s[sk_idx]["act_bias"][i])



        elif self.ga_form == "mx":
            targets = np.array([p.raw_target for p in self.pairs], float)

            for pair_list, pair_inv_list in self.groups_cycles:
                dev = np.abs(np.prod(targets[pair_list]) / np.prod(targets[pair_inv_list]) - 1)
                cd = self.HU_decay(dev)
                
                for i in pair_list+pair_inv_list:
                    self.pairs[i].cycle_decay = min(self.pairs[i].cycle_decay, cd) # penalize pairs in cycle
            
            # Optimization: force values to obey cycles (use previous chunk decision)
            if False:
                targets = self.OPT_mx(targets)
                
            # update action multipliers
            for i, pair in enumerate(self.pairs):
                s = self.sym_transf_args
                target = targets[i]

                pair.target_m = target

                # importance of pair
                diff = pair.get_target_spread(float(target))
                pair.func_weight = self.HF(diff)

                mean_abs_out = np.mean(np.abs(pair.out)) # free variable
                pair.heterogeneity = np.mean(np.abs(pair.inp*target-pair.out)) / mean_abs_out

                if target == 0: # If the target is zero, it's a bad target 
                    continue 

                update_w = self.HU_coef * pair.cycle_decay
                for sk_idx in pair.sym_idx_01_dir: # 0->1 direct
                    s[sk_idx]["act_multiplier"][pair.ID[1]] += update_w * (target - s[sk_idx]["act_multiplier"][pair.ID[1]])
                    s[sk_idx]["func_weight"][pair.ID[1]] = pair.func_weight

                for sk_idx in pair.sym_idx_01_inv: # 0->1 inverse
                    s[sk_idx]["act_multiplier"][pair.ID[1]] += update_w * (-target - s[sk_idx]["act_multiplier"][pair.ID[1]])
                    s[sk_idx]["func_weight"][pair.ID[1]] = pair.func_weight

                for sk_idx in pair.sym_idx_10_dir: # 1->0 direct
                    s[sk_idx]["act_multiplier"][pair.ID[0]] += update_w * (1/target - s[sk_idx]["act_multiplier"][pair.ID[0]])
                    s[sk_idx]["func_weight"][pair.ID[0]] = pair.func_weight

                for sk_idx in pair.sym_idx_10_inv: # 1->0 inverse
                    s[sk_idx]["act_multiplier"][pair.ID[0]] += update_w * (-1/target - s[sk_idx]["act_multiplier"][pair.ID[0]])
                    s[sk_idx]["func_weight"][pair.ID[0]] = pair.func_weight



    def during_optimization(self, indices, rollout_data, sym_losses, clip_range):

        ppo = self.ppo
        total_sym_loss = 0


        with th.no_grad():
            mu_last_of_obs = ppo.policy(rollout_data.observations, deterministic=True)[0].detach().clone()

        stdev_last = th.exp( ppo.policy.log_std.detach().clone() ) # stdev snapshot right before mini-batch update
        v_last_of_obs = ppo.policy.predict_values(rollout_data.observations).detach().clone().flatten() # NN prediction for mean state values

        for sk in self.sym_transf_args:

            if sk["algorithm"] == "none":
                sym_losses.append(0) 
                continue

            sym_obs = sk["sym_obs"][indices]
            v_of_sym_obs = ppo.policy.predict_values(sym_obs).flatten() # used for value loss

            features = ppo.policy.extract_features(sym_obs)
            latent_pi = ppo.policy.mlp_extractor.forward_actor(features)
            mu_of_sym_obs = ppo.policy.action_net(latent_pi)

            #============================================== different algorithms
            if sk["algorithm"] == "asl":
                if sk["use_value_gate"]: # add value diff binary gate
                    v_last_of_sym_obs = v_of_sym_obs.detach().clone() # used for value gate
                    higher_v_last_of_obs = sk["k_v_alpha"]*v_last_of_obs + sk["k_v_minus_alpha"] * th.abs(v_last_of_obs)
                    value_gate = th.clamp(th.sign(higher_v_last_of_obs - v_last_of_sym_obs), min=0) # gate only prevents large mistakes
                else:
                    value_gate = 1

                dead_zone_gate = sk["dead_zone_gate"][indices]
                mu_old_of_sym_obs = sk["mu_old_of_sym_obs"][indices]

                sym_mu_last_of_obs = th.multiply( th.index_select(mu_last_of_obs, 1, sk["act_indices"]), sk["act_multiplier"]) + sk["act_bias"]
                tau = th.clip(sym_mu_last_of_obs, mu_old_of_sym_obs - sk["delta_mu"], mu_old_of_sym_obs + sk["delta_mu"])

                sym_ratio_1 = sk["func_weight"]*(th.square(tau - mu_old_of_sym_obs) - th.square(tau - mu_of_sym_obs)) / (2 * stdev_last * stdev_last)
                sym_ratio_2 = th.exp( th.sum( sym_ratio_1 , 1))

                sym_loss = -sk["policy_weight"] * (sym_ratio_2 * dead_zone_gate * value_gate).mean()
                sym_losses.append(sym_loss.item()) # register before adding value loss
                sym_loss += sk["value_weight"] * (th.square(rollout_data.returns - v_of_sym_obs) * dead_zone_gate * value_gate).mean()

            elif sk["algorithm"] == "psl":
                '''
                Proximal symmetry loss, as the original paper (uses main coefficient and value coefficient)
                '''
                pi_old_sym = sk["pi_old_of_sym_obs_and_sym_mu_old_of_obs"][indices]
                sym_mean_action_old = sk["sym_mu_old_of_obs"][indices]
                    
                pi_sym = ppo.policy.get_distribution(sym_obs).log_prob(sym_mean_action_old) # pi( g(am) | f(s) )    "pi_of_sym_obs_and_sym_mu_old_of_obs"
                psl_ratio = th.exp(th.min( pi_sym, rollout_data.old_log_prob ) - pi_old_sym)

                sym_loss = -sk["policy_weight"] * th.min(psl_ratio, th.tensor(1 + clip_range)  ).mean()
                sym_losses.append(sym_loss.item()) # register before adding value loss
                sym_loss += sk["value_weight"] * F.mse_loss(rollout_data.returns, v_of_sym_obs)

            elif sk["algorithm"] == "msl_original":
                '''
                Mirror symmetry loss, also as the original paper uses (main coefficient) and after generalization uses (value coefficient)
                '''
                features = ppo.policy.extract_features(rollout_data.observations)
                latent_pi = ppo.policy.mlp_extractor.forward_actor(features)
                mu_of_obs = ppo.policy.action_net(latent_pi) # mean action prediction

                sym_of_mu_of_sym_obs = th.multiply( th.index_select(mu_of_sym_obs, 1, sk["act_indices"]), sk["act_multiplier"])
                
                sym_loss = sk["policy_weight"] * th.square(mu_of_obs - sym_of_mu_of_sym_obs).mean()
                sym_losses.append(sym_loss.item()) # register before adding value loss
                sym_loss += sk["value_weight"] * F.mse_loss(rollout_data.returns, v_of_sym_obs)

            elif sk["algorithm"] == "msl":
                '''
                Mirror symmetry loss, also as the original paper uses (main coefficient) and after generalization uses (value coefficient)
                Fixed for rotation symmetry
                '''
                features = ppo.policy.extract_features(rollout_data.observations)
                latent_pi = ppo.policy.mlp_extractor.forward_actor(features)
                mu_of_obs = ppo.policy.action_net(latent_pi) # mean action prediction

                sym_mu_of_obs = th.multiply( th.index_select(mu_of_obs, 1, sk["act_indices"]), sk["act_multiplier"])
                
                sym_loss = sk["policy_weight"] * th.square(sym_mu_of_obs - mu_of_sym_obs).mean()
                sym_losses.append(sym_loss.item()) # register before adding value loss
                sym_loss += sk["value_weight"] * F.mse_loss(rollout_data.returns, v_of_sym_obs)

            else:
                raise NotImplementedError
            
            total_sym_loss += sym_loss

        return total_sym_loss

    def log(self):
        if self.symmetry_log is not None:
            if self.sym_log_countdown == 0:
                self.sym_log_countdown = self.symmetry_log_freq - 1
                with open(self.symmetry_log, 'a+') as f:
                    writer = csv.writer(f)
                    cols = [self.ppo.num_timesteps, th.exp(self.ppo.policy.log_std).mean().item()]
                    for sk in self.sym_transf_args: cols.append(sk["v_dist"])
                    for sk in self.sym_transf_args: cols.append(sk["dead_ratio"])
                    for sk in self.sym_transf_args: cols.append(sk["value_weight"])
                    cols.extend( [p.target for p in self.singles.values()] ) # target
                    cols.extend( [p.target_m for p in self.pairs] ) # target
                    cols.extend( [p.target_b for p in self.pairs] ) # target
                    cols.extend( [p.cycle_decay for p in self.pairs] ) # weight
                    cols.extend( [p.heterogeneity for p in self.pairs] ) # heterogeneity
                    cols.extend( [p.func_weight for p in self.pairs] ) # dimension weight for each component of the pair
                    for sk in self.sym_transf_args: cols.extend(sk["act_multiplier"].tolist())
                    for sk in self.sym_transf_args: cols.extend(sk["act_bias"].tolist())
                    writer.writerow(cols)
            else:
                self.sym_log_countdown -= 1


    def OPT_mx(self, targets):
        def func(x):
            l = gnp.mean(gnp.array([gnp.mean(gnp.square(p.inp*x[i]-p.out)) for i,p in enumerate(self.pairs)]))
            a = gnp.mean(gnp.array([gnp.abs(gnp.prod(x[pair_list]) / gnp.prod(x[pair_inv_list]) - 1) for pair_list, pair_inv_list in self.groups_cycles]))
            return l + 10 * a

        func_der=jacobian(func)

        def func_stats(x):
            l = np.mean(np.array([np.mean(np.square(p.inp*x[i]-p.out)) for i,p in enumerate(self.pairs)]))
            a = np.mean(np.array([np.abs(np.prod(x[pair_list]) / np.prod(x[pair_inv_list]) - 1) for pair_list, pair_inv_list in self.groups_cycles]))
            print("main loss:      ", l)
            print("constraint loss:", a)
            print("total loss:     ", l + 10 * a)

        from time import time
        t1 = time()
        res = minimize(func, targets, method='BFGS', jac=func_der, options={'disp': False})
        t2 = time()

        func_stats(targets)
        print("before:", targets)
        print("after: ", res.x)
        func_stats(res.x)
        print("t=",int((t2-t1)*1000), "ms\n")
        return res.x

    def OPT_mxb(self, targets):
        def func(x):
            l = gnp.mean(gnp.array([gnp.mean(gnp.square(p.inp*x[i*2]+x[i*2+1]-p.out)) for i,p in enumerate(self.pairs)]))
            devs_sum = 0
            for pair_list, pair_inv_list in self.groups_cycles:
                dev = 1
                for i in pair_list:
                    m,b = x[i*2:i*2+2]
                    dev = dev * m + b
                for i in pair_inv_list:    
                    m,b = x[i*2:i*2+2]
                    dev = (dev - b) / (m+1e-8)
                devs_sum += gnp.abs(dev - 1)
            a = devs_sum/len(self.groups_cycles)

            return l + 10 * a

        func_der=jacobian(func)

        def func_stats(x):
            l = np.mean(np.array([np.mean(np.square(p.inp*x[i*2]+x[i*2+1]-p.out)) for i,p in enumerate(self.pairs)]))
            devs_sum = 0
            for pair_list, pair_inv_list in self.groups_cycles:
                dev = 1
                for i in pair_list:
                    m,b = x[i*2:i*2+2]
                    dev = dev * m + b
                for i in pair_inv_list:    
                    m,b = x[i*2:i*2+2]
                    dev = (dev - b) / (m+1e-8)
                devs_sum += np.abs(dev - 1)
            a = devs_sum/len(self.groups_cycles)
            print("main loss:      ", l)
            print("constraint loss:", a)
            print("total loss:     ", l + 10 * a)

        from time import time
        t1 = time()
        res = minimize(func, targets, method='BFGS', jac=func_der, options={'disp': False})
        t2 = time()

        func_stats(targets)
        print("before:", targets)
        print("after: ", res.x)
        func_stats(res.x)
        print("t=",int((t2-t1)*1000), "ms\n")
        return res.x


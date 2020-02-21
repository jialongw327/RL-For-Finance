import sys 
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')

from typing import Tuple, Sequence, Mapping
from MDP_For_RL_Tabular import MDPRLTabular
from Markov_Functions import A, S, SAf, get_rv_gen_func_single
import numpy as np

class MonteCarlo():
    
    def __init__(
        self,
        mdp_rep_for_rl: MDPRLTabular,
        first_visit: bool,
        num_episodes: int,
        max_steps: int
    ) -> None:
        self.mdp_rep: MDPRLTabular = mdp_rep_for_rl
        self.first_visit = first_visit
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps
        
    def get_mc_path(self, pol: SAf, start_state: S) -> Sequence[Tuple[S, A, float, bool]]:
        path = []
        state = start_state
        steps = 0
        occ_states = []
        action_dic = {s: get_rv_gen_func_single(pol[s]) \
                      for s in self.mdp_rep.state_action_dict.keys()}
        
        while state not in self.mdp_rep.terminal_states \
                        and steps <= self.max_steps:
            first_time = state not in occ_states
            occ_states.append(state)
            action = action_dic[state]()
            next_state, reward = \
                       self.mdp_rep.state_reward_gen_dict[state][action]()
            path.append((state, action, reward, first_time))
            steps += 1
            state = next_state
        
        path.append((state, 'End',0,'End'))
            
        return path
    
    def get_value_func_dict(self, pol: SAf) -> Mapping[S, float]:
        counts_dict = {s: 0 for s in self.mdp_rep.state_action_dict.keys()}
        vf_dict = {s: 0.0 for s in self.mdp_rep.state_action_dict.keys()}
        episodes = 0
        
        while episodes < self.num_episodes:
            start_state = self.mdp_rep.init_state_gen()
            mc_path = self.get_mc_path(pol, start_state)
            rew_arr = np.array([x for _, _, x, _ in mc_path[:-1]])
            if mc_path[-1][0] in self.mdp_rep.terminal_states:
                returns = self.get_returns(rew_arr)
            else:
                raise RuntimeError('MC path did not reach terminal states, \
                                    increase max_steps')
            for i, r in enumerate(returns):
                s, _, _, f = mc_path[i]
                if not self.first_visit or f:
                    counts_dict[s] += 1
                    c = counts_dict[s]
                    vf_dict[s] = (vf_dict[s] * (c - 1) + r) / c
            episodes += 1
        
        return vf_dict
        
    def get_returns(self, reward: Sequence[float]) -> list:
        T = len(reward)
        discount = np.power(self.mdp_rep.gamma, np.arange(T))
        return [np.dot(discount[:T-i],reward[i:]) for i in range(T)]
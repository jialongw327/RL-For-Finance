import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

import numpy as np
from MP import MP
from MRP import MRP
from typing import Generic, Set, Mapping

from Markov_Functions import find_all_states, find_actions_for_states
from Markov_Functions import find_lean_transitions, 
from Markov_Functions import mdp_rep_to_mrp_rep1, mdp_rep_to_mrp_rep2
from Generic_TypeVars import S, A
from Standard_TypeVars import SSf, STSff, SATSff
from General_Utils import zip_dict_of_tuple


class MDP(Generic[S, A]):

    def __init__(self, mdp_graph: SATSff, gamma: float):
        d = {k: zip_dict_of_tuple(v) for k, v in mdp_graph.items()}
        mdp_transition_graph, mdp_reward_graph = zip_dict_of_tuple(d)
        # Find all the states in the mdp-graph -> stores in a set {state}
        self.all_states: Set[S] = find_all_states(mdp_graph)
        # Find actions for each state -> stores in a dict {state: {action}}
        self.state_action_dict: Mapping[S, Set[A]] = \
                find_actions_for_states(mdp_graph)
        # Find the transition graph without reward {state: {action: {state, prob}}}         
        self.transitions: Mapping[S, Mapping[A, Mapping[S, float]]] = \
                {s: {a: find_lean_transitions(v1) for a, v1 in v.items()}
                 for s, v in mdp_transition_graph.items()}     
        print(self.transitions)        

    def find_sink_states(self) -> Set[S]:
        '''Sink states' rewards don't have to be 0!'''
        # A sink state must go back to itself only
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }
        
    def find_terminal_states(self) -> Set[S]:
        # Terminal states are sink states with reward = 0
        sink = self.get_sink_states()
        return {s for s in sink if
                all(is_approx_eq(r, 0.0) for _, r in self.rewards[s].items())}
        
    def find_mrp(self, pol: Policy) -> MRP:
        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew = mdp_rep_to_mrp_rep2(self.rewards, pol.policy_data)
        return MRP({s: (v, rew[s]) for s, v in tr.items()}, self.gamma)
        
    def find_value_func_dict(self, pol: Policy)\
            -> Mapping[S, float]:
        # Find v_pi(s)
        mrp_obj = self.get_mrp(pol)
        value_func_vec = mrp_obj.get_value_func_vec()
        nt_vf = {mrp_obj.nt_states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.nt_states_list))}
        t_vf = {s: 0. for s in self.terminal_states}
        return {**nt_vf, **t_vf}
                
    def find_act_value_func_dict(self, pol: Policy)\
            -> Mapping[S, Mapping[A, float]]:
        # Find q_pi(s, a) 
        v_dict = self.get_value_func_dict(pol)
        return {s: {a: r + self.gamma * sum(p * v_dict[s1] for s1, p in
                                            self.transitions[s][a].items())
                    for a, r in v.items()}
                for s, v in self.rewards.items()}

    
    def find_improved_policy(self, pol: Policy) -> DetPolicy:
        # Find the policy that maximizes the act-value function (greedy)
        q_dict = self.find_act_value_func_dict(pol)
        return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                          for s, v in q_dict.items()})
            
    def find_optimal_policy(self, tol=1e-4) -> DetPolicy:
        # Find the optimal policy using policy iteration
        pol = Policy({s: {a: 1. / len(v) for a in v} for s, v in
                      self.state_action_dict.items()})
        vf = self.get_value_func_dict(pol)
        epsilon = tol * 1e4
        while epsilon >= tol:
            pol = self.find_improved_policy(pol)
            new_vf = self.find_value_func_dict(pol)
            epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
            vf = new_vf
        return pol
        
    
if __name__ == '__main__':
    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'a': ({3: 1.0}, 0.0),
            'b': ({3: 1.0}, 0.0)
        }
    }
        
        
    student = {
            'C1': {
                'Study': ({'C2': 1}, -2.0),
                'FB':({'Facebook': 1}, -1.0)
                    },
            'C2':{'Study': ({'C3': 1}, -2.0),
                  'SLP':({'Sleep': 1}, 0.0)
                    },
            'C3':{'Study':({'Sleep': 1}, 0.0),
                  'Pub':({'C1': 0.2,'C2': 0.4 ,'C3': 0.4}, 1.0)
                    },
            'Facebook':{'FB':({'Facebook': 1}, -1.0), 'Quit':({'C1': 1}, 0.0)
                    },
            'Sleep':{'FB':({'Sleep': 1}, 0.0),
                     'Study':({'Sleep': 1}, 0.0),
                     'Pub':({'Sleep': 1}, 0.0),
                     'SLP':({'Sleep': 1}, 0.0)}
            }
            
    mdp_obj = MDP(student, 0.95)
import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

import numpy as np

from typing import Generic, Set, Mapping, Sequence
from operator import itemgetter

from Generic_TypeVars import S, A
from Standard_TypeVars import SATSff
from General_Utils import zip_dict_of_tuple, is_approx_eq

from MP import MP
from MRP import MRP
from Markov_Functions import find_all_states, find_actions_for_states
from Markov_Functions import mdp_rep_to_mrp_rep1, mdp_rep_to_mrp_rep2


from Policy import Policy
from DetPolicy import DetPolicy

class MDP(Generic[S, A]):

    def __init__(self, mdp_graph: SATSff, gamma: float):
        
        d = {k: zip_dict_of_tuple(v) for k, v in mdp_graph.items()}
        
        mdp_transition_graph, mdp_reward_graph = zip_dict_of_tuple(d)
        
        # Find the transition graph without reward {state: {action: {state, prob}}}         
        self.transitions = mdp_transition_graph
        
        self.rewards: Mapping[S, Mapping[A, float]] = mdp_reward_graph
        
        # Find all the states in the mdp-graph -> stores in a set {state}
        
        self.terminal_states: Set[S] = self.find_terminal_states()
        
        self.all_states: Set[S] = find_all_states(mdp_graph)
        
        self.nt_states: Set[S] = self.find_nt_states_list()
        
        # Find actions for each state -> stores in a dict {state: {action}}
        self.state_action_dict: Mapping[S, Set[A]] = \
                find_actions_for_states(mdp_graph)
                
        self.gamma: float = gamma
        
        self.terminal_states: Set[S] = self.find_terminal_states()

    def find_sink_states(self) -> Set[S]:
        '''Sink states' rewards don't have to be 0!'''
        # A sink state must go back to itself only
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }
        
    def find_terminal_states(self) -> Set[S]:
        # Terminal states are sink states with reward = 0
        sink = self.find_sink_states()
        return {s for s in sink if
                all(is_approx_eq(r, 0.0) for _, r in self.rewards[s].items())}
        
    def find_nt_states_list(self) -> Sequence[S]:
        return [s for s in self.all_states
                if s not in self.terminal_states]
           
        
    def find_mrp(self, pol: Policy) -> MRP:
        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew = mdp_rep_to_mrp_rep2(self.rewards, pol.policy_data)
        return MRP({s: (v, rew[s]) for s, v in tr.items()}, self.gamma)
        
    def find_value_func_dict(self, pol: Policy)\
            -> Mapping[S, float]:
        # Find v_pi(s)
        mrp_obj = self.find_mrp(pol)
        value_func_vec = mrp_obj.find_value_function_vector()
        nt_vf = {mrp_obj.nt_states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.nt_states_list))}
        t_vf = {s: 0. for s in self.terminal_states}
        return {**nt_vf, **t_vf}
                
    def find_act_value_func_dict(self, pol: Policy)\
            -> Mapping[S, Mapping[A, float]]:
        # Find q_pi(s, a) 
        v_dict = self.find_value_func_dict(pol)
        return {s: {a: r + self.gamma * sum(p * v_dict[s1] for s1, p in
                                            self.transitions[s][a].items())
                    for a, r in v.items()}
                for s, v in self.rewards.items()}
    
    def policy_evaluation(self, pol:Policy) -> np.array:
        '''Iterative way to find the value functions given a specific policy'''
        mrp = self.find_mrp(pol)
        v0 = np.zeros(len(self.nt_states))
        print(mrp.trans_matrix)
        print(mrp.reward_vector)
        converge = False
        while not converge:
            v1 = mrp.reward_vector + self.gamma*mrp.trans_matrix.dot(v0)
            converge = is_approx_eq(np.linalg.norm(v1), np.linalg.norm(v0))
            v0 = v1
        return v1
    
    def find_improved_policy(self, pol: Policy) -> DetPolicy:
        '''Find the policy that maximizes the action-value function 
        in one iteration (greedy)'''
        q_dict = self.find_act_value_func_dict(pol)
        return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                          for s, v in q_dict.items()})
            
    def policy_iteration(self, tol=1e-4) -> DetPolicy:
        ''' Find the optimal policy using policy iteration '''
        pol = Policy({s: {a: 1. / len(v) for a in v} for s, v in
                      self.state_action_dict.items()})
        vf = self.find_value_func_dict(pol)
        epsilon = tol * 1e4
        while epsilon >= tol:
            pol = self.find_improved_policy(pol)
            new_vf = self.find_value_func_dict(pol)
            epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
            vf = new_vf
        return pol
        
    
    def value_iteration(self, tol = 1e-4) -> DetPolicy:
        return ValueError
    
if __name__ == '__main__':
#    data = {
#        1: {
#            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
#            'b': ({2: 0.3, 3: 0.7}, 2.8),
#            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
#        },
#        2: {
#            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
#            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
#        },
#        3: {
#            'a': ({3: 1.0}, 0.0),
#            'b': ({3: 1.0}, 0.0)
#        }
#    }
#        
        
    student = { 
            'C1': {
                'Study': ({'C2': 1}, -2.0),
                'FB':({'Facebook': 1}, -1.0)
                    },
            
            'C2':{'Study': ({'C3': 1}, -2.0),
                  'SLP':({'Sleep': 1}, 0.0)
                    },
            
            'C3':{'Study':({'Sleep': 1}, 10.0),
                  'Pub':({'C1': 0.2,'C2': 0.4 ,'C3': 0.4}, 1.0)
                    },
            
            'Facebook':{'FB':({'Facebook': 1}, -1.0), 'Quit':({'C1': 1}, 0.0)
                    },
            
            'Sleep':{'SLP':({'Sleep': 1}, 0.0)}
            }
            
    policy = {
            
            'C1': {
                'Study': 0.5,
                'FB':0.5
                    },
                    
            'C2':{'Study': 0.5,
                  'SLP':0.5
                    },
                  
            'C3':{'Study':0.5,
                  'Pub':0.5
                    },
                  
            'Facebook':{'FB':0.5,
                        'Quit':0.5
                    },
            'Sleep':{
                     'SLP':1
                    }
            }

        
    mdp_obj = MDP(student, 0.999999)
    pol_obj = Policy(policy)
    mrp_obj = mdp_obj.find_mrp(pol_obj)
    
    
    #print('The sink states are: \n',mdp_obj.find_sink_states(), "\n")
    #print('The terminal states are: \n',mdp_obj.find_terminal_states(), "\n")
    print('The value obtained from pol evaluation is: \n',mdp_obj.policy_evaluation(pol_obj), "\n")
    print('The value obtained from pol evaluation is: \n',mdp_obj.find_value_func_dict(pol_obj), "\n")
    pol_obj_vi = print(mdp_obj.policy_iteration())
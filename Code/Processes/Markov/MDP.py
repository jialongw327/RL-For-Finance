import numpy as np
from MP import MP
from MRP import MRP
from typing import Generic, Set
import sys
sys.path.append('Code/Utils')
sys.path.append('Code/Processes/Markov')
from Markov_Functions import find_all_states
from Generic_TypeVars import S, A
from Standard_TypeVars import SSf, STSff, SATSff
from General_Utils import zip_dict_of_tuple


class MDP(Generic[S, A]):

    def __init__(self, mdp_graph: SATSff, gamma: float):
        d = {k: zip_dict_of_tuple(v) for k, v in mdp_graph.items()}
        mdp_transition_graph, mdp_reward_graph = zip_dict_of_tuple(d)
        self.all_states: Set[S] = find_all_states(mdp_graph)
        print(mdp_reward_graph)



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
### MRP add two more properties on MP
#in this section, the default reward function based on david silver definition
import numpy as np
from MP import MP
import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
from typing import Mapping, Sequence, Set
from Generic_TypeVars import S
from Standard_TypeVars import STSff
from General_Utils import zip_dict_of_tuple, is_approx_eq

class MRP(MP):
    
    def __init__(self, mrp_graph:STSff, gamma:float):
        # Super Class from MP
        mp_graph, reward_graph = zip_dict_of_tuple(mrp_graph)
        self.mygraph = mp_graph
        super().__init__(mp_graph)                                        
        self.reward_graph: Mapping[S, float] = reward_graph
        # Gamma is the discount factor
        self.gamma: float = gamma
        self.reward_vector=self.find_reward_vector()
        self.terminal_states = self.find_terminal_states()
        self.not_terminal_states_list: Sequence[S] = self.find_nt_states_list()     
        
    def find_reward(self) -> Mapping[S, float]:
        return self.reward_graph

    def find_reward_vector(self) -> np.array:
        return np.array([self.reward_graph[i] for i in self.state])
    
    def find_value_function_vector(self) -> np.array:
        # Bellman Equation in Matrix Form
        # Blows up when gamma = 1
        return np.linalg.inv(np.eye(len(self.state)) - self.gamma * self.transition_matrix
        ).dot(self.find_reward_vector())
        
    def find_terminal_states(self) -> Set[S]:
        sink = self.find_sink_states()
        return {s for s in sink if is_approx_eq(self.reward_graph[s], 0.0)}
            
    def find_nt_states_list(self) -> Sequence[S]:
        return [s for s in self.all_states_list
                if s not in self.terminal_states]
        
    

if __name__ == '__main__':
    
    student = {'Facebook':({'C1': 0.1, 'Facebook': 0.9}, -1.0),
               'C1':({'C2': 0.5, 'Facebook': 0.5}, -2.0),
               'C2':({'C3': 0.8, 'Sleep': 0.2}, -2.0),
               'C3':({'Pass': 0.6, 'Pub': 0.4}, -2.0),
               'Pass':({'Sleep': 1.0}, 10.0),
               'Pub':({'C1': 0.2, 'C2': 0.4, 'C3': 0.4}, 1.0),
               'Sleep':({'Sleep': 1.0}, 0.0)}
    gamma = 0.9
    test_mrp = MRP(student, gamma)
    print('Get value function vector:\n',test_mrp.find_value_function_vector(),'\n')
    print('Find reward vector:\n', test_mrp.find_reward_vector())   
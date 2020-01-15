### MRP add two more properties on MP
#in this section, the default reward function based on david silver definition
import numpy as np
from MP import MP
from typing import Mapping,Sequence,TypeVar

State = TypeVar('S')

class MRP(MP):
    
    def __init__(self,graph:Mapping[State,Mapping[State, float]],reward:Mapping[State, float], gamma:float):
        # Super Class from MP
        super().__init__(graph)                                        
        self.reward_graph: Mapping[State, float] = reward
        # Gamma is the discount factor
        self.gamma: float = gamma
        self.reward_vector=self.find_reward_vector()
        
    def find_reward(self) -> Mapping[State, float]:
        return self.reward_graph

    def find_reward_vector(self) -> np.array:
        return np.array([self.reward_graph[i] for i in self.state])
        
    def find_value_function_vector(self) -> np.array:
        # Bellman Equation in Matrix Form
        # Blows up when gamma = 1
        return np.linalg.inv(np.eye(len(self.state)) - self.gamma * self.transition_matrix
        ).dot(self.find_reward_vector())


if __name__ == '__main__':
    student = {'C1':{'C2': 0.5, 'Facebook': 0.5},
               'C2':{'C3': 0.8, 'Sleep': 0.2},
               'C3':{'Pass': 0.6, 'Pub': 0.4},
               'Pass':{'Sleep': 1.0},
               'Pub':{'C1': 0.2, 'C2': 0.4, 'C3': 0.4},
               'Facebook':{'C1': 0.1, 'Facebook': 0.9},
               'Sleep':{'Sleep': 1.0}}
    reward = {'C1': -2.0, 'C2': -2.0, 'C3': -2.0, 'Pass': 10.0,\
              'Pub': 1.0, 'Facebook': -1.0, 'Sleep': 0.0}
    gamma = 0.9
    test_mrp = MRP(student, reward, gamma)
    print('Get value function vector:\n',test_mrp.find_value_function_vector(),'\n')
    print('Find reward vector:\n', test_mrp.find_reward_vector())   
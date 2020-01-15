import numpy as np
from scipy.linalg import eig
from typing import Mapping,Sequence,TypeVar
import matplotlib.pyplot as plt

State = TypeVar('State')

class MP:
    # The Mapping type indicates a dict[state, dict[state, number]]
    # self.transition_graph is a dictionary
    def __init__(self, graph: Mapping[State, Mapping[State, float]]):
        self.state: Sequence[State] = list(graph.keys())
        self.transition_graph: Mapping[State, Mapping[State, float]] = graph
        self.transition_matrix: np.array = self.find_transition_matrix()
    
    def check_mp(self) -> bool:
        graph = self.transition_graph
        states = set(graph.keys())
        # Check the successor states is the subset of current states
        b1 = set().union(*list(graph.values())).issubset(states)
        # Check the probabilities are positive
        b2 = all(all(p >= 0 for p in graph[state].values()) for state in graph)
        # Check the sum of probabilities for each state equals to 1
        b3 = all((sum(graph[state].values())== 1) for state in  graph)        
        return b1 and b2 and b3    
    
    def find_transition_matrix(self)->np.array:
        '''Transition Matrix'''
        num: int = len(self.state)
        transition_matrix = np.zeros((num,num))
        for i in range(num):
            s=self.state[i]
            for j in range(num):
                s_prime=self.state[j]
                if s_prime in self.transition_graph[s].keys():
                    transition_matrix[i,j]=self.transition_graph[s][s_prime]
        return transition_matrix    
   
    def find_transition_probability(self,cur_state:State,next_state:State)->float:
        '''Get specific transition Probabilities between two states'''
        if cur_state in self.state and next_state in self.state:
            if next_state in self.transition_graph[cur_state].keys():
                
                return self.transition_graph[cur_state][next_state]
            else:
                return 0.0
        else: 
            raise ValueError
            
    def find_stationary_distribution(self) -> Mapping[State, Mapping[State, float]]:
        '''Stationary Probabilities'''
        # We use the eigen-value approach
        eig_val, eig_vec_right = eig(self.transition_matrix.T)
        eig_vec_left = eig_vec_right[:,np.isclose(eig_val, 1)]
        eig_vec = eig_vec_left[:,0]
        stationary = eig_vec/eig_vec.sum()
        stationary = stationary.real
        return {s: stationary[i] for i, s in enumerate(self.state)}
    
    def check_stationary_distribution(self) -> np.array:
        return NotImplementedError
    
    def find_transition_graph(self) -> Mapping[State, Mapping[State, float]]:
        '''Current Graph'''
        return self.transition_graph
    
    def find_all_states(self) -> Sequence[State]:
        '''Current State'''
        return self.state    
    
    def find_terminal_states(self) -> Sequence[State]:
        return NotImplementedError


if __name__=='__main__':
    transition_graph = {
        'Sunny': {'Sunny': 0.1, 'Cloudy': 0.6, 'Rainy': 0.1, 'Snowy': 0.2},
        'Cloudy': {'Sunny': 0.25, 'Cloudy': 0.22, 'Rainy': 0.24, 'Snowy': 0.29},
        'Rainy': {'Sunny': 0.7, 'Cloudy': 0.3},
        'Snowy': {'Sunny': 0.3, 'Cloudy': 0.5, 'Rainy': 0.2}
    }
    student = {'C1':{'C2': 0.5, 'Facebook': 0.5},
               'C2':{'C3': 0.8, 'Sleep': 0.2},
               'C3':{'Pass': 0.6, 'Pub': 0.4},
               'Pass':{'Sleep': 1.0},
               'Pub':{'C1': 0.2, 'C2': 0.4, 'C3': 0.4},
               'Facebook':{'C1': 0.1, 'Facebook': 0.9},
               'Sleep':{'Sleep': 1.0}}

    #test_mp=MP(transition_graph)
    test_mp=MP(student)
    print(test_mp.check_mp(), '\n')
    print('Find transition matrix :\n',test_mp.find_transition_matrix(),'\n')
    print('Find stationary distribution:\n',test_mp.find_stationary_distribution(),'\n')
    print('Find all states:\n',test_mp.find_all_states(),'\n')
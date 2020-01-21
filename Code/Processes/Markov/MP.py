import numpy as np
from scipy.linalg import eig
from typing import Mapping,Sequence,Set
from Markov_Functions import verify_mp
import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
from Generic_TypeVars import S
from Standard_TypeVars import SSf

class MP:
    # The Mapping type indicates a dict[state, dict[state, number]]
    # self.transition_graph is a dictionary
    def __init__(self, graph: SSf):
        if verify_mp(graph):
            self.state: Sequence[S] = list(graph.keys())
            self.transition_graph: Mapping[S, Mapping[S, float]] = graph
            self.transition_matrix: np.array = self.find_transition_matrix()
        else:
            raise ValueError
    
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
   
    def find_transition_probability(self,cur_state:S,next_state:S)->float:
        '''Get specific transition Probabilities between two states'''
        if cur_state in self.state and next_state in self.state:
            if next_state in self.transition_graph[cur_state].keys():
                
                return self.transition_graph[cur_state][next_state]
            else:
                return 0.0
        else: 
            raise ValueError
            
    def find_stationary_distribution(self) -> Mapping[S, Mapping[S, float]]:
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
    
    def find_transition_graph(self) -> Mapping[S, Mapping[S, float]]:
        '''Current Graph'''
        return self.transition_graph
    
    def find_all_states(self) -> Sequence[S]:
        '''Current State'''
        return self.state    
    
    def find_sink_states(self) -> Set[S]:
        return {k for k, v in self.transition_graph.items()
                if len(v) == 1 and k in v.keys()}

if __name__=='__main__':

    student = {'C1':{'C2': 0.5, 'Facebook': 0.5},
               'C2':{'C3': 0.8, 'Sleep': 0.2},
               'C3':{'Pass': 0.6, 'Pub': 0.4},
               'Pass':{'Sleep': 1.0},
               'Pub':{'C1': 0.2, 'C2': 0.4, 'C3': 0.4},
               'Facebook':{'C1': 0.1, 'Facebook': 0.9},
               'Sleep':{'Sleep': 1.0}}

    #test_mp=MP(transition_graph)
    test_mp=MP(student)
    print('Find transition matrix :\n',test_mp.find_transition_matrix(),'\n')
    print('Find stationary distribution:\n',test_mp.find_stationary_distribution(),'\n')
    print('Find all states:\n',test_mp.find_all_states(),'\n')
    print('Find sink state:\n',test_mp.find_sink_states(),'\n')
import sys 
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
from typing import Mapping, Set, Callable, Tuple, Generic

from Markov_Functions import get_rv_gen_func_single

from Generic_TypeVars import S, A

Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]


class MDPRepForRLTabular(Generic[S, A]):

    def __init__(
        self,
        state_action_dict: Mapping[S, Set[A]],
        terminal_states: Set[S],
        state_reward_gen_dict: Type1,
        gamma: float
    ) -> None:
        self.state_action_dict: Mapping[S, Set[A]] = state_action_dict
        self.terminal_states: Set[S] = terminal_states
        self.state_reward_gen_dict: Type1 = state_reward_gen_dict
        self.gamma = gamma
    
    def init_state_gen(self):
        return {s: 1. / len(self.state_action_dict) for s
                 in self.state_action_dict.keys()}
        
    def init_state_action_gen(self):
        return {(s, a): 1. / sum(len(v) for v
                                  in self.state_action_dict.values())
                 for s, v1 in self.state_action_dict.items() for a in v1}
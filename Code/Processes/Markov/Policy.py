import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

from typing import Mapping, Generic, Dict

from Markov_Functions import verify_policy
from Markov_Functions import find_epsilon_action_probs
from Markov_Functions import find_softmax_action_probs
from Generic_TypeVars import S, A


class Policy(Generic[S, A]):

    def __init__(self, data: Dict[S, Mapping[A, float]]) -> None:
        if verify_policy(data):
            self.policy_data = data
        else:
            raise ValueError

    def find_state_probabilities(self, state: S) -> Mapping[A, float]:
        return self.policy_data[state]

    def find_state_action_probability(self, state: S, action: A) -> float:
        return self.get_state_probabilities(state).get(action, 0.)

    def find_state_action_to_epsilon_greedy(
        self,
        state: S,
        action_value_dict: Mapping[A, float],
        epsilon: float
    ) -> None:
        self.policy_data[state] = find_epsilon_action_probs(
            action_value_dict,
            epsilon
        )

    def edit_state_action_to_softmax(
            self,
            state: S,
            action_value_dict: Mapping[A, float]
    ) -> None:
        self.policy_data[state] = find_softmax_action_probs(
            action_value_dict
        )

    def __repr__(self):
        return self.policy_data.__repr__()

    def __str__(self):
        return self.policy_data.__str__()
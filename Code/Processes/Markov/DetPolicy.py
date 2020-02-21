import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

from typing import Mapping

from Generic_TypeVars import S, A

from Policy import Policy

class DetPolicy(Policy):

    def __init__(self, det_policy_data: Mapping[S, A]) -> None:
        super().__init__({s: {a: 1.0} for s, a in det_policy_data.items()})

    def find_action_for_state(self, state: S) -> A:
        return list(self.find_state_probabilities(state).keys())[0]

    def find_state_to_action_map(self) -> Mapping[S, A]:
        return {s: self.find_action_for_state(s) for s in self.policy_data}

    def __repr__(self) -> str:
        return self.find_state_to_action_map().__repr__()

    def __str__(self) -> str:
        return self.find_state_to_action_map().__str__()
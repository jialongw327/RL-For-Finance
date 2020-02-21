import sys 
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')

from typing import Tuple

from MRP_Refined import MRPRefined
from MDP import MDP
from Policy import Policy
from Standard_TypeVars import SASf, SAf, SASTff
from General_Utils import zip_dict_of_tuple, merge_dicts
from Markov_Functions import flatten_sasf_dict, mdp_rep_to_mrp_rep1, unflatten_sasf_dict
from Markov_Functions import flatten_ssf_dict, unflatten_ssf_dict

import numpy as np


class MDPRefined(MDP):

    def __init__(
        self,
        info: SASTff,
        gamma: float
    ) -> None:
        d1, d2, d3 = MDPRefined.split_info(info)
        super().__init__(
            {s: {a: (v1, d3[s][a]) for a, v1 in v.items()}
             for s, v in d1.items()},
            gamma
        )
        self.rewards_refined: SASf = d2

    @staticmethod
    def split_info(info: SASTff) -> Tuple[SASf, SASf, SAf]:
        c = {s: {a: zip_dict_of_tuple(v1) for a, v1 in v.items()}
             for s, v in info.items()}
        d = {k: zip_dict_of_tuple(v) for k, v in c.items()}
        d1, d2 = zip_dict_of_tuple(d)
        d3 = {s: {a: sum(np.prod(x) for x in v1.values())
                  for a, v1 in v.items()} for s, v in info.items()}
        return d1, d2, d3

    def get_mrp_refined(self, pol: Policy) -> MRPRefined:
        flat_transitions = flatten_sasf_dict(self.transitions)
        flat_rewards_refined = flatten_sasf_dict(self.rewards_refined)

        flat_exp_rewards = merge_dicts(flat_rewards_refined, flat_transitions, lambda x, y: x * y)
        exp_rewards = unflatten_sasf_dict(flat_exp_rewards)

        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew_ref = mdp_rep_to_mrp_rep1(
            exp_rewards,
            pol.policy_data
        )
        flat_tr = flatten_ssf_dict(tr)
        flat_rew_ref = flatten_ssf_dict(rew_ref)
        flat_norm_rewards = merge_dicts(flat_rew_ref, flat_tr, lambda x, y: x / y)
        norm_rewards = unflatten_ssf_dict(flat_norm_rewards)

        return MRPRefined(
            {s: {s1: (v1, norm_rewards[s][s1]) for s1, v1 in v.items()}
             for s, v in tr.items()},
            self.gamma
        )
        
    

if __name__ == '__main__':
    # data = {
    #     1: {
    #         'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
    #         'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
    #         'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
    #     },
    #     2: {
    #         'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
    #         'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
    #     },
    #     3: {
    #         'a': {3: (1.0, 0.0)},
    #         'b': {3: (1.0, 0.0)}
    #     }
    # }
    # mdp_refined_obj = MDPRefined(data, 0.95)
    # print(mdp_refined_obj.all_states)
    # print(mdp_refined_obj.transitions)
    # print(mdp_refined_obj.rewards)
    # print(mdp_refined_obj.rewards_refined)
    # terminal = mdp_refined_obj.get_terminal_states()
    # print(terminal)

    print("This is MDPRefined")
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    mdp_refined_obj = MDPRefined(mdp_refined_data, 0.97)
    print("Transitions")
    print(mdp_refined_obj.transitions)
    print("Rewards Refined")
    print(mdp_refined_obj.rewards_refined)

    print("----------------")
    print("This is the Policy")
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)
    print(pol_obj.policy_data)

    print("----------------")
    print("This is MRPRefined")
    mrp_refined_obj = mdp_refined_obj.get_mrp_refined(pol_obj)
    print("Transitions")
    print(mrp_refined_obj.mpgraph)
    print("Rewards Refined")
    print(mrp_refined_obj.rewards_refined)

    print("-----------------")
    print("This is MDP")
    print("Rewards")
    print(mdp_refined_obj.rewards)

    print("-----------------")
    print("This is MRP from MDP")
    mrp_obj1 = mdp_refined_obj.find_mrp(pol_obj)
    print("Rewards")
    print(mrp_obj1.reward_graph)

    print("---------------")
    print("This is MRP from MRPRefined")
    print("Rewards")
    print(mrp_refined_obj.reward_graph)


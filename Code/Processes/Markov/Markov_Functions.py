import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

import numpy as np
from typing import Mapping, Sequence, Set, Any
from operator import itemgetter


from Generic_TypeVars import S, A
from Standard_TypeVars import SSf, SATSff, SASf, SAf
from General_Utils import memoize, is_approx_eq, sum_dicts

@memoize
def find_all_states(d: Mapping[S, Any]) -> Set[S]:
    temp = list(d.keys())
    lookup = set()  # a temporary lookup set
    result = [x for x in temp if x not in lookup and lookup.add(x) is None]
    return result


@memoize
def find_actions_for_states(mdp_data: Mapping[S, Mapping[A, Any]])\
        -> Mapping[S, Set[A]]:
    return {k: set(v.keys()) for k, v in mdp_data.items()}


@memoize
def find_all_actions(mdp_data: Mapping[S, Mapping[A, Any]]) -> Set[A]:
    return set().union(*find_actions_for_states(mdp_data).values())


def find_lean_transitions(d: Mapping[S, float]) -> Mapping[S, float]:
    return {s: v for s, v in d.items() if not is_approx_eq(v, 0.0)}


def verify_transitions(
    states: Set[S],
    tr_seq: Sequence[Mapping[S, float]]
) -> bool:
    b1 = set().union(*tr_seq).issubset(states)
    b2 = all(all(x >= 0 for x in d.values()) for d in tr_seq)
    b3 = all(is_approx_eq(sum(d.values()), 1.0) for d in tr_seq)
    return b1 and b2 and b3


@memoize
def verify_mp(mp_data: SSf) -> bool:
    all_st = find_all_states(mp_data)
    val_seq = list(mp_data.values())
    return verify_transitions(all_st, val_seq)


@memoize
def verify_mdp(mdp_data: SATSff) -> bool:
    all_st = find_all_states(mdp_data)
    check_actions = all(len(v) > 0 for _, v in mdp_data.items())
    val_seq = [v2 for _, v1 in mdp_data.items() for _, (v2, _) in v1.items()]
    return verify_transitions(all_st, val_seq) and check_actions

@memoize
def verify_policy(policy_data: SAf) -> bool:
    return all(is_approx_eq(sum(v.values()), 1.0) for s, v in policy_data.items())


def mdp_rep_to_mrp_rep1(
    mdp_rep: SASf,
    policy_rep: SAf
) -> SSf:
    return {s: sum_dicts([{s1: p * v2 for s1, v2 in v[a].items()}
                          for a, p in policy_rep[s].items()])
            for s, v in mdp_rep.items()}

def mdp_rep_to_mrp_rep2(
    mdp_rep: SAf,
    policy_rep: SAf
) -> Mapping[S, float]:
    return {s: sum(p * v[a] for a, p in policy_rep[s].items())
            for s, v in mdp_rep.items()}

def find_epsilon_action_probs(
    action_value_dict: Mapping[A, float],
    epsilon: float
) -> Mapping[A, float]:
    max_act = max(action_value_dict.items(), key=itemgetter(1))[0]
    if epsilon == 0:
        ret = {max_act: 1.}
    else:
        ret = {a: epsilon / len(action_value_dict) +
               (1. - epsilon if a == max_act else 0.)
               for a in action_value_dict.keys()}
    return ret


def find_softmax_action_probs(
    action_value_dict: Mapping[A, float]
) -> Mapping[A, float]:
    aq = {a: q - max(action_value_dict.values())
          for a, q in action_value_dict.items()}
    exp_sum = sum(np.exp(q) for q in aq.values())
    return {a: np.exp(q) / exp_sum for a, q in aq.items()}

#import numpy as np
from typing import Mapping, Sequence, Set, Any
import sys
sys.path.append('Code/Utils')
from Generic_TypeVars import S, A
from Standard_TypeVars import SSf, SATSff
from General_Utils import memoize, is_approx_eq

@memoize
def find_all_states(d: Mapping[S, Any]) -> Set[S]:
    return set(d.keys())


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



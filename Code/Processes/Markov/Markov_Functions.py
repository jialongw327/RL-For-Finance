import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')

import numpy as np
from typing import Mapping, Sequence, Set, Any, Tuple, Callable
from operator import itemgetter


from Generic_TypeVars import S, A
from Standard_TypeVars import SSf, SATSff, SASf, SAf, SASTff
from General_Utils import memoize, is_approx_eq, sum_dicts, FlattenedDict
from scipy.stats import rv_discrete

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


def find_transition_matrix(tr: SSf) -> np.ndarray:
    s_list = find_all_states(tr)
    mat = np.zeros((len(s_list), len(s_list)))
    for s_i in s_list:
        for s_j in list(tr[s_i].keys()):
            mat[s_list.index(s_i),s_list.index(s_j)] = tr[s_i][s_j]
    return mat

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
def verify_mdp_2(mdp_data: SASTff) -> bool:
    row_sum = [sum(list(sf.values())) for dic in mdp_data.values() for sf in dic.values()]
    for each_row in row_sum:
        if np.abs(each_row - 1) > 1e-8:
            return False
    return True

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


def find_split_MDP_graph(mdp_graph: SASTff) -> Tuple[SASf, SASf, SAf]:
    tr = {s: {a: {s1: v2[0] for s1,v2 in v1.items()} for a,v1 in v.items()} 
                for s,v in mdp_graph.items()}
    rr = {s: {a: {s1: v2[1] for s1,v2 in v1.items()} for a,v1 in v.items()} 
                for s,v in mdp_graph.items()}
    state_reward = {s: {a: sum([v2[0]*v2[1] for s1,v2 in v1.items()]) for a,v1 in v.items()} 
                for s,v in mdp_graph.items()}
    return tr, rr, state_reward

def flatten_sasf_dict(sasf: SASf) -> FlattenedDict:
    return [((s, a, s1), f)
            for s, asf in sasf.items()
            for a, sf in asf.items()
            for s1, f in sf.items()]
    
def flatten_ssf_dict(ssf: SSf) -> FlattenedDict:
    return [((s, s1), f)
            for s, sf in ssf.items()
            for s1, f in sf.items()]
    
def unflatten_sasf_dict(q: FlattenedDict) -> SASf:
    dsasf = {}
    for (sas, f) in q:
        dasf = dsasf.get(sas[0], {})
        dsf = dasf.get(sas[1], {})
        dsf[sas[2]] = f
        dasf[sas[1]] = dsf
        dsasf[sas[0]] = dasf
    return dsasf

def unflatten_ssf_dict(q: FlattenedDict) -> SSf:
    dssf = {}
    for (ss, f) in q:
        dsf = dssf.get(ss[0], {})
        dsf[ss[1]] = f
        dssf[ss[0]] = dsf
    return dssf

def get_rv_gen_func_single(prob_dict: Mapping[Any, float])\
        -> Callable[[], S]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    
    return lambda rvd=rvd, outcomes=outcomes: outcomes[rvd.rvs(size=1)[0]]

def get_state_reward_gen_func(
    prob_dict: Mapping[S, float],
    rew_dict: Mapping[S, float]
) -> Callable[[], Tuple[S, float]]:
    gf = get_rv_gen_func_single(prob_dict)
    
    def ret_func(gf=gf, rew_dict=rew_dict) -> Tuple[S, float]:
        state_outcome = gf()
        reward_outcome = rew_dict[state_outcome]
        return state_outcome, reward_outcome

    return ret_func

def get_state_reward_gen_dict(tr: SASf, rr: SASf) \
        -> Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]:
    return {s: {a: get_state_reward_gen_func(tr[s][a], rr[s][a])
                for a, _ in v.items()}
            for s, v in rr.items()}
        
def find_expected_action_value(action_qv: Mapping[A, float], epsilon: float) -> float:
    _, val_opt = max(action_qv.items(), key = lambda l:l[1])
    m = len(action_qv.keys())
    return sum([val*epsilon/m for val in action_qv.values()]) \
               + val_opt * (1 - epsilon)
               
def find_epsilon_greedy_action(action_qv: Mapping[A, float], epsilon: float) -> float:
    action_opt, val_opt = max(action_qv.items(), key = lambda l:l[1])
    m = len(action_qv.keys())
    prob_dict = {a: epsilon/m + 1 - epsilon if a == action_opt else epsilon/m\
                 for a,v in action_qv.items()}
    gf = get_rv_gen_func_single(prob_dict)
    
    return gf()
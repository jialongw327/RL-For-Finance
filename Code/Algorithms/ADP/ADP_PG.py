import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')

from typing import Callable, Sequence, Tuple
import numpy as np

from Function_Approximation_Specification import FuncApproxSpec
from Function_Approximation_Base import FuncApproxBase

from Helper_Functions import get_policy_as_action_dict

from Generic_Typevars import S, A
from Standard_TypeVars import VFType, QFType
from Standard_TypeVars import PolicyType, PolicyActDictType
from Optimization_Base import OptBase
from MDP_Rep_For_ADP_PG import MDPRepForADPPG


class ADPPolicyGradient(OptBase):
    """
    This class is meant for continuous state spaces and continuous action
    spaces. Here we blend ADP (i.e., DP with function approximation for
    value function) with PG (i.e., gradient of policy drives policy
    improvement). See the class ADP to understand how plain ADP is done
    (i.e., without PG) - this class is an extension/variant of that
    technique where the policy improvement is policy gradient instead
    of argmax. Note that both ADP and this class are model-based
    methods, meaning we have access to state transition probabilities
    and expected rewards, and we take advantage of that information
    in the algorithm. On the other hand, the class PolicyGradient is
    a model-free algorithm based on PolicyGradient (also meant for
    continuous states/actions).
    """

    def __init__(
        self,
        mdp_rep_for_adp_pg: MDPRepForADPPG,
        reinforce: bool,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        max_steps: int,
        actor_lambda: float,
        critic_lambda: float,
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]],
        vf_fa_spec: FuncApproxSpec,
        pol_fa_spec: Sequence[FuncApproxSpec]

    ) -> None:
        self.mdp_rep: MDPRepForADPPG = mdp_rep_for_adp_pg
        self.reinforce: bool = reinforce
        self.num_state_samples: int = num_state_samples
        self.num_next_state_samples: int = num_next_state_samples
        self.num_action_samples: int = num_action_samples
        self.num_batches: int = num_batches
        self.max_steps: int = max_steps
        self.actor_lambda: float = actor_lambda
        self.critic_lambda: float = critic_lambda
        self.score_func: Callable[[A, Sequence[float]], Sequence[float]] =\
            score_func
        self.sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]] =\
            sample_actions_gen_func
        self.vf_fa: FuncApproxBase = vf_fa_spec.get_vf_func_approx_obj()
        self.pol_fa: Sequence[FuncApproxBase] =\
            [s.get_vf_func_approx_obj() for s in pol_fa_spec]
            
            
    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        mo = self.mdp_rep
        sr_func = self.mdp_rep.state_reward_gen_func
        # rew_func = mdp_func_to_mrp_func2(self.mdp_rep.reward_func, polf)
        # prob_func = mdp_func_to_mrp_func1(self.mdp_rep.transitions_func, polf)
        init_samples_func = self.mdp_rep.init_states_gen_func
        for _ in range(self.num_batches):
            samples = init_samples_func(self.num_state_samples)
            values = [sum(ap * (r + mo.gamma * self.vf_fa.get_func_eval(s1))
                          for a, ap in polf(s).items() for s1, r in
                          sr_func(s, a, self.num_next_state_samples)) /
                      self.num_next_state_samples for s in samples]
            avg_grad = [g / len(samples) for g in
                        self.vf_fa.get_sum_loss_gradient(samples, values)]
            self.vf_fa.update_params_from_gradient(avg_grad)
            # print(self.vf_fa.get_func_eval(1))
            # print(self.vf_fa.get_func_eval(2))
            # print(self.vf_fa.get_func_eval(3))
            # print("-----")

        return self.vf_fa.get_func_eval 
    
    def get_act_value_func_fa(self, polf: PolicyActDictType) -> QFType:
        v_func = self.get_value_func_fa(polf)

        # noinspection PyShadowingNames
        def state_func(s: S, v_func=v_func) -> Callable[[A], float]:

            # noinspection PyShadowingNames
            def act_func(a: A, v_func=v_func) -> float:
                return sum(r + self.mdp_rep.gamma * v_func(s1) for
                           s1, r in self.mdp_rep.state_reward_gen_func(
                    s,
                    a,
                    self.num_next_state_samples
                )
                           ) / self.num_next_state_samples

            return act_func

        return state_func

    def get_value_func(self, pol_func: PolicyType) -> VFType:
        return self.get_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                self.num_action_samples
            )
        )

    def get_act_value_func(self, pol_func: PolicyType) -> QFType:
        return self.get_act_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                self.num_action_samples
            )
        )

    def get_policy_as_policy_type(self) -> PolicyType:

        def pol(s: S) -> Callable[[int], Sequence[A]]:

            # noinspection PyShadowingNames
            def gen_func(samples: int, s=s) -> Sequence[A]:
                return self.sample_actions_gen_func(
                    [f.get_func_eval(s) for f in self.pol_fa],
                    samples
                )

            return gen_func

        return pol

    def get_path(
        self,
        start_state: S
    ) -> Sequence[Tuple[S, Sequence[float], A]]:
        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            pdf_params = [f.get_func_eval(state) for f in self.pol_fa]
            action = self.sample_actions_gen_func(pdf_params, 1)[0]
            res.append((
                state,
                pdf_params,
                action
            ))
            steps += 1
            terminate = steps >= self.max_steps or\
                self.mdp_rep.terminal_state_func(state)
            state = self.mdp_rep.state_reward_gen_func(state, action, 1)[0][0]
        return res

    def get_optimal_reinforce_func(self) -> PolicyType:
        mo = self.mdp_rep
        init_samples_func = mo.init_states_gen_func
        sc_func = self.score_func

        for _ in range(self.num_batches):
            init_states = init_samples_func(self.num_state_samples)
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for init_state in init_states:
                states = []
                disc_return_scores = []
                return_val = 0.
                this_path = self.get_path(init_state)

                for i, (s, pp, a) in enumerate(this_path[::-1]):
                    i1 = len(this_path) - 1 - i
                    states.append(s)
                    reward = sum(r for _, r in mo.state_reward_gen_func(
                        s,
                        a,
                        self.num_next_state_samples
                    )) / self.num_next_state_samples
                    return_val = return_val * mo.gamma + reward
                    disc_return_scores.append(
                        [return_val * mo.gamma ** i1 * x for x in sc_func(a, pp)]
                    )
                    # print(s)
                    # print(pp)

                pg_arr = np.vstack(disc_return_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_sum_objective_gradient(
                        states,
                        - pg_arr[:, i]
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            # print("--------")
            for i, pp_fa in enumerate(self.pol_fa):
                gradient = [pg / self.num_state_samples for pg in pol_grads[i]]
                # print(gradient)
                pp_fa.update_params_from_gradient(gradient)

        return self.get_policy_as_policy_type()

    def get_optimal_tdl_func(self) -> PolicyType:
        mo = self.mdp_rep
        init_samples_func = mo.init_states_gen_func
        sc_func = self.score_func
        for _ in range(self.num_batches):
            init_states = init_samples_func(self.num_state_samples)
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for init_state in init_states:
                gamma_pow = 1.
                states = []
                deltas = []
                disc_scores = []
                this_path = self.get_path(init_state)

                for s, pp, a in this_path:
                    target = sum(r + mo.gamma * self.vf_fa.get_func_eval(s1)
                                 for s1, r in mo.state_reward_gen_func(
                        s,
                        a,
                        self.num_next_state_samples
                    )) / self.num_next_state_samples
                    delta = target - self.vf_fa.get_func_eval(s)
                    states.append(s)
                    deltas.append(delta)
                    disc_scores.append(
                        [gamma_pow * x for x in sc_func(a, pp)]
                    )
                    gamma_pow *= mo.gamma
                    # print(s)
                    # print(pp)
                    # print(a)
                    # print(target)

                # print(list(zip(states, deltas)))

                self.vf_fa.update_params_from_gradient(
                    self.vf_fa.get_el_tr_sum_objective_gradient(
                        states,
                        np.power(mo.gamma, np.arange(len(this_path))),
                        - np.array(deltas),
                        mo.gamma * self.critic_lambda
                    )
                )

                pg_arr = np.vstack(disc_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_el_tr_sum_objective_gradient(
                        states,
                        pg_arr[:, i],
                        - np.array(deltas),
                        mo.gamma * self.actor_lambda
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            for i, pp_fa in enumerate(self.pol_fa):
                gradient = [pg / self.num_state_samples for pg in pol_grads[i]]
                # print(gradient)
                pp_fa.update_params_from_gradient(gradient)

            # print(self.vf_fa.get_func_eval(1))
            # print(self.vf_fa.get_func_eval(2))
            # print(self.vf_fa.get_func_eval(3))
            # print("----")

        return self.get_policy_as_policy_type()

    def get_optimal_stoch_policy_func(self) -> PolicyType:
        return self.get_optimal_reinforce_func() if self.reinforce\
            else self.get_optimal_tdl_func()

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        papt = self.get_optimal_stoch_policy_func()

        def opt_det_pol_func(s: S) -> A:
            return tuple(np.mean(
                papt(s)(self.num_action_samples),
                axis=0
            ))

        return opt_det_pol_func
        
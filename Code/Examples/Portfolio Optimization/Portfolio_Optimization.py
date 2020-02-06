import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Utils')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms\\ADP')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms\\DP')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Functions')

from typing import Tuple, Sequence, Callable
import numpy as np
from Beta_Distribution import BetaDistribution

from DNN_Spec import DNNSpec
from Function_Approximation_Specification import FuncApproxSpec
from MDP_Rep_For_ADP_PG import MDPRepForADPPG
from MDP_Rep_For_RL_PG import MDPRepForRLPG

from ADP_PG import ADPPolicyGradient
from PG import PolicyGradient

StateType = Tuple[int, float]
ActionType = Tuple[float, ...]


class PortfolioOptimization:

    SMALL_POS = 1e-8

    def __init__(
        self,
        num_risky: int,
        riskless_returns: Sequence[float],
        returns_gen_funcs: Sequence[Callable[[int], np.ndarray]],
        cons_util_func: Callable[[float], float],
        beq_util_func: Callable[[float], float],
        discount_rate: float
    ) -> None:
        if PortfolioOptimization.validate_spec(
            num_risky,
            riskless_returns,
            returns_gen_funcs,
            discount_rate
        ):
            self.num_risky: int = num_risky
            self.riskless_returns: Sequence[float] = riskless_returns
            self.epochs: int = len(riskless_returns)
            self.returns_gen_funcs: Sequence[Callable[[int], np.ndarray]]\
                = returns_gen_funcs
            self.cons_util_func: Callable[[float], float] = cons_util_func
            self.beq_util_func: Callable[[float], float] = beq_util_func
            self.discount_rate: float = discount_rate
        else:
            raise ValueError
        
               
    @staticmethod
    def validate_spec(
        num_risky_assets: int,
        riskless_returns_seq: Sequence[float],
        returns_gen: Sequence[Callable[[int], np.ndarray]],
        disc_fact: float
    ) -> bool:
        # Make sure we have risky asset
        check_1 = num_risky_assets >= 1
        # Riskless returns >= 0
        check_2 = all(x > 0 for x in riskless_returns_seq)
        check_3 = len(riskless_returns_seq) == len(returns_gen)
        # The discount factor >=0 && <=1
        check_4 = 0. <= disc_fact <= 1.
        return all([check_1, check_2, check_3, check_4])
    
    # Epoch t is from time t to time (t+1), for 0 <= t < T
    # where T = number of epochs. At time T (i.e., at end of epoch
    # (T-1)), the process ends and W_T is the quantity of bequest.
    # The order of operations in an epoch t (0 <= t < T) is:
    # 1) Observe the state (t, W_t).
    # 2) Consume C_t . W_t so wealth drops to W_t . (1 - C_t)
    # 3) Allocate W_t . (1 - C_t) to n risky assets and 1 riskless asset
    # 4) Riskless asset grows by r_t and risky assets grow stochastically
    #    with wealth growing from W_t . (1 - C_t) to W_{t+1}
    # 5) At the end of final epoch (T-1) (i.e., at time T), bequest W_T.
    #
    # U_Beq(W_T) is the utility of bequest, U_Cons(W_t) is the utility of
    # consmption.
    # State at the start of epoch t is (t, W_t)
    # Action upon observation of state (t, W_t) is (C_t, A_1, .. A_n)
    # where 0 <= C_t <= 1 is the consumption and A_i, i = 1 .. n, is the
    # allocation to risky assets. Allocation to riskless asset will be set to
    # A_0 = 1 - \sum_{i=1}^n A_i. If stochastic return of risky asset i is R_i:
    # W_{t+1} = W_t . (1 - C_t) . (A_0 . (1+r) + \sum_{i=1}^n A_i . (1 + R_i))

    @staticmethod
    def score_func(
        action: ActionType,
        params: Sequence[float]
    ) -> Sequence[float]:
        """
        :param action: is a tuple (a_0, a_1, ...., a_n) where a1
        is the consumption, and a_i, 1 <= i <= n, is the allocation
        for risky asset i.
        :param params: is a Sequence (mu, nu, mu_1, ..., mu_n,
        sigma^2_1,..., sigma^2_n) (of length 2n + 2) where (mu, nu) describes the
        beta distribution for the consumption 0 < a_0 < 1, and (mu_i, sigma^2_i)
        describes the normal distribution for the allocation a_i of risky asset
        i, 1 <= i <= n.
        :return: nabla_{params} [log_e p(a_0, a_1, ..., a_n; mu, nu,
        mu_1, ..., mu_n, sigma^2_1, ..., sigma^2_n)], which is a Sequence of
        length 2n+2
        """
        n = len(action) - 1
        mu, nu = params[:2]
        means = params[2:2+n]
        variances = params[2+n:]
        cons = action[0]
        alloc = action[1:]
        
        # mu and nu draw from Beta Distribution 
        mu_score, nu_score = BetaDistribution(mu, nu).get_mu_nu_scores(cons)
        
        # mean and variance draw from Gaussian Distribution
        means_score = [(alloc[i] - means[i]) / v for i, v in
                       enumerate(variances)]
        variances_score = [-0.5 * (1. / v - means_score[i] ** 2) for i, v in
                           enumerate(variances)]
        return [mu_score, nu_score] + means_score + variances_score
from typing import Tuple, Sequence, Callable
import numpy as np





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
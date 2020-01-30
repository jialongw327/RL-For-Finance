from typing import NamedTuple, Callable, Mapping, Tuple
import numpy as np




class MertonPortfolio(NamedTuple):
    expiry: float  # = T
    rho: float  # = discount rate
    r: float  # = risk-free rate
    mu: np.ndarray  # = risky rate means (1-D array of length num risky assets)
    cov: np.ndarray  # = risky rate covariances (2-D square array of length num risky assets)
    epsilon: float  # = bequest parameter
    gamma: float  # = CRRA parameter

    SMALL_POS = 1e-8


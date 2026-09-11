"""
2D Ackley function — the fitness (objective) function to be minimized.

    f(x1, x2) = -a * exp(-b * sqrt(0.5 * (x1^2 + x2^2)))
                - exp(0.5 * (cos(c*x1) + cos(c*x2)))
                + a + exp(1)

Recommended parameters (from assignment instructions): a=20, b=0.2, c=2*pi
Global minimum: f(0, 0) = 0
Search domain: -5 <= x1, x2 <= 5
"""

import numpy as np

A = 20.0
B = 0.2
C = 2 * np.pi

BOUNDS = (-5.0, 5.0)


def ackley(x1: float, x2: float, a: float = A, b: float = B, c: float = C) -> float:
    """Evaluate the 2D Ackley function at a single point (x1, x2)."""
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
    return term1 + term2 + a + np.exp(1)


def ackley_vec(
    population: np.ndarray, a: float = A, b: float = B, c: float = C
) -> np.ndarray:
    """
    Evaluate the 2D Ackley function for a whole population at once.

    population: ndarray of shape (N, 2), each row is a chromosome [x1, x2]
    returns:    ndarray of shape (N,) with the fitness (function value) of each chromosome
    """
    x1 = population[:, 0]
    x2 = population[:, 1]
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
    return term1 + term2 + a + np.exp(1)

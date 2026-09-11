"""
Small sanity-check test suite.

Run with:  pytest test_ga.py -v
"""

import numpy as np
import pytest
from ackley import ackley, ackley_vec
from genetic_algorithm import GAConfig, GeneticAlgorithm


def test_ackley_global_minimum_is_zero_at_origin():
    assert ackley(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_ackley_is_positive_away_from_origin():
    assert ackley(3.0, -2.0) > 0


def test_ackley_vec_matches_scalar():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [-2.5, 4.0]])
    vec_result = ackley_vec(pts)
    scalar_result = np.array([ackley(x, y) for x, y in pts])
    assert np.allclose(vec_result, scalar_result)


def test_ga_improves_over_generations():
    """The best-so-far fitness history should be non-increasing thanks to elitism."""
    cfg = GAConfig(generations=30, seed=1)
    ga = GeneticAlgorithm(config=cfg)
    result = ga.run()
    history = result.history_best
    assert all(history[i + 1] <= history[i] + 1e-12 for i in range(len(history) - 1))


def test_ga_converges_close_to_zero():
    """With the recommended settings the GA should get reasonably close to 0."""
    cfg = GAConfig(generations=100, seed=7)
    ga = GeneticAlgorithm(config=cfg)
    result = ga.run()
    assert result.best_fitness < 1.0  # generous bound; typical runs land well under 0.1


def test_chromosomes_stay_within_bounds():
    cfg = GAConfig(generations=50, seed=3)
    ga = GeneticAlgorithm(config=cfg)
    result = ga.run()
    low, high = cfg.bounds
    assert low <= result.best_chromosome[0] <= high
    assert low <= result.best_chromosome[1] <= high

"""
Genetic Algorithm (GA) to find the global minimum of the 2D Ackley function.

Implements exactly the operators/settings specified in the assignment:

    Chromosome        : [x1, x2]  (real-value encoding, each gene in [-5, 5])
    Population size    : 50
    Generations        : 100 (or until convergence)
    Crossover prob (Pc): 0.8
    Mutation prob (Pm) : 0.05
    Elitism            : top 1 individual carried over unchanged
    Selection          : Roulette Wheel
    Crossover          : 1-point crossover
    Mutation           : 1-point gene mutation (Gaussian perturbation of one gene)
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from ackley import BOUNDS, ackley_vec


@dataclass
class GAConfig:
    pop_size: int = 50
    generations: int = 100
    pc: float = 0.8  # crossover probability
    pm: float = 0.05  # mutation probability
    elitism: int = 1  # number of elite individuals carried over unchanged
    bounds: tuple = BOUNDS  # (low, high) applied to every gene
    mutation_sigma: float = 0.5  # std-dev of the Gaussian mutation delta
    convergence_tol: float = 1e-6  # stop early if best fitness <= this
    seed: int | None = None


@dataclass
class GAResult:
    best_chromosome: np.ndarray
    best_fitness: float
    history_best: list[float] = field(
        default_factory=list
    )  # best fitness per generation
    history_mean: list[float] = field(
        default_factory=list
    )  # mean fitness per generation
    generations_run: int = 0
    converged: bool = False


class GeneticAlgorithm:
    """A small, readable GA engine, specialized for minimizing a 2D real-valued fitness function."""

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], np.ndarray] = ackley_vec,
        config: GAConfig = None,
    ):
        self.fitness_fn = fitness_fn
        self.cfg = config or GAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ---------------------------------------------------------------- init
    def _init_population(self) -> np.ndarray:
        low, high = self.cfg.bounds
        return self.rng.uniform(low, high, size=(self.cfg.pop_size, 2))

    # ---------------------------------------------------------- selection
    def _roulette_wheel_select(
        self, population: np.ndarray, fitness: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Roulette Wheel (fitness-proportionate) selection.

        We are MINIMIZING the Ackley function, so raw fitness values can't be used
        directly as selection weights (lower is better, but roulette wheel needs
        "bigger slice = more likely"). We invert the fitness by subtracting every
        value from (worst + a small epsilon), so the best (lowest Ackley value)
        individual gets the largest slice of the wheel.
        """
        worst = fitness.max()
        # small epsilon avoids every individual getting exactly zero weight
        # when they all tie with the worst value.
        inverted = (worst - fitness) + 1e-8
        probabilities = inverted / inverted.sum()
        idx = self.rng.choice(len(population), size=n, replace=True, p=probabilities)
        return population[idx]

    # ----------------------------------------------------------- crossover
    def _one_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """1-point crossover over a 2-gene chromosome [x1, x2].

        The only interior cut point for a 2-gene chromosome is between gene 0
        and gene 1, i.e. child1 = [p1.x1, p2.x2] and child2 = [p2.x1, p1.x2].
        """
        if self.rng.random() < self.cfg.pc:
            child1 = np.array([parent1[0], parent2[1]])
            child2 = np.array([parent2[0], parent1[1]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    # ------------------------------------------------------------ mutation
    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """1-point gene mutation: pick ONE of the two genes and perturb it with a
        Gaussian-distributed random delta, then clip back into bounds."""
        if self.rng.random() < self.cfg.pm:
            gene_idx = self.rng.integers(0, 2)
            delta = self.rng.normal(0, self.cfg.mutation_sigma)
            chromosome = chromosome.copy()
            chromosome[gene_idx] += delta
            low, high = self.cfg.bounds
            chromosome[gene_idx] = np.clip(chromosome[gene_idx], low, high)
        return chromosome

    # --------------------------------------------------------------- run
    def run(self, verbose: bool = False) -> GAResult:
        cfg = self.cfg
        population = self._init_population()
        fitness = self.fitness_fn(population)

        history_best, history_mean = [], []
        best_chromosome = population[np.argmin(fitness)].copy()
        best_fitness = fitness.min()
        converged = False
        gen = 0

        for gen in range(1, cfg.generations + 1):
            # 1. Sort ascending (best/lowest Ackley value first)
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            # 2. Elitism: set aside the best `elitism` individuals, unchanged
            elite = population[: cfg.elitism].copy()

            # 3. Selection + 4. Crossover + 5. Mutation -> build offspring
            offspring = []
            while len(offspring) < cfg.pop_size - cfg.elitism:
                p1, p2 = self._roulette_wheel_select(population, fitness, 2)
                c1, c2 = self._one_point_crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(c1)
                if len(offspring) < cfg.pop_size - cfg.elitism:
                    offspring.append(c2)

            # 6. Form next generation: elite + offspring
            population = np.vstack([elite, np.array(offspring)])
            fitness = self.fitness_fn(population)

            # Track best-so-far (elitism guarantees monotonic improvement)
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()

            history_best.append(best_fitness)
            history_mean.append(fitness.mean())

            if verbose and (gen % 10 == 0 or gen == 1):
                print(
                    f"Gen {gen:3d} | best f = {best_fitness:.6f} | "
                    f"x* = ({best_chromosome[0]: .4f}, {best_chromosome[1]: .4f}) | "
                    f"mean f = {fitness.mean():.4f}"
                )

            # 7. Stopping criteria: close enough to global minimum (0)
            if best_fitness <= cfg.convergence_tol:
                converged = True
                break

        return GAResult(
            best_chromosome=best_chromosome,
            best_fitness=float(best_fitness),
            history_best=history_best,
            history_mean=history_mean,
            generations_run=gen,
            converged=converged,
        )

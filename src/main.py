"""
Entry point for the GA assignment.

Runs the Genetic Algorithm on the 2D Ackley function using the settings
recommended in the assignment instructions, prints a generation-by-generation
convergence log, saves plots (convergence curve + solution on the Ackley
surface) to src/output/, and finally runs a short classical-optimizer
comparison (as requested in the instructions) plus a reduced-generation
GA run (20/25 generations) for reference.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ackley import BOUNDS, ackley, ackley_vec
from genetic_algorithm import GAConfig, GeneticAlgorithm
from scipy.optimize import minimize

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_main_ga(seed: int = 42):
    print("=" * 70)
    print("Genetic Algorithm — 2D Ackley Function Global Minimum")
    print("=" * 70)
    cfg = GAConfig(seed=seed)
    ga = GeneticAlgorithm(fitness_fn=ackley_vec, config=cfg)
    result = ga.run(verbose=True)

    print("-" * 70)
    print(
        f"Finished after {result.generations_run} generation(s) "
        f"({'converged' if result.converged else 'reached max generations'})"
    )
    print(
        f"Best solution found: x1 = {result.best_chromosome[0]:.6f}, "
        f"x2 = {result.best_chromosome[1]:.6f}"
    )
    print(
        f"Best fitness f(x1, x2) = {result.best_fitness:.8f}  (global minimum is 0 at (0,0))"
    )
    print("=" * 70)
    return result, cfg


def plot_convergence(result, path):
    plt.figure(figsize=(8, 5))
    plt.plot(result.history_best, label="Best fitness (so far)", linewidth=2)
    plt.plot(result.history_mean, label="Mean population fitness", alpha=0.6)
    plt.axhline(
        0, color="gray", linestyle="--", linewidth=1, label="Global minimum (0)"
    )
    plt.xlabel("Generation")
    plt.ylabel("Ackley fitness value")
    plt.title("GA Convergence on the 2D Ackley Function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_surface_with_solution(result, path):
    low, high = BOUNDS
    grid = np.linspace(low, high, 250)
    X1, X2 = np.meshgrid(grid, grid)
    Z = ackley(X1, X2)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(X1, X2, Z, levels=50, cmap="viridis")
    fig.colorbar(contour, label="f(x1, x2)")
    ax.plot(
        0,
        0,
        marker="*",
        color="white",
        markersize=16,
        label="True global minimum (0,0)",
    )
    ax.plot(
        result.best_chromosome[0],
        result.best_chromosome[1],
        marker="o",
        color="red",
        markersize=10,
        label="GA best solution",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("2D Ackley Function — GA Solution vs True Minimum")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def classical_comparison():
    """Compare the GA result against a classical (gradient/derivative-free)
    optimizer, as requested in the instructions."""
    print("\nClassical optimizer comparison (scipy.optimize.minimize, Nelder-Mead)")
    x0 = np.array(
        [4.0, -4.0]
    )  # start far from the minimum, same as a random GA individual might
    res = minimize(lambda x: ackley(x[0], x[1]), x0, method="Nelder-Mead")
    print(f"  Start point:       x0 = ({x0[0]}, {x0[1]})")
    print(
        f"  Classical result:  x* = ({res.x[0]:.6f}, {res.x[1]:.6f}), f(x*) = {res.fun:.8f}"
    )
    return {"start": x0.tolist(), "x_star": res.x.tolist(), "f_star": float(res.fun)}


def reduced_generation_ga_runs(seed: int = 42):
    """Run the GA with 20 and 25 generations (as suggested in the instructions)
    to see how solution quality scales with generation count."""
    print("\nReduced-generation GA runs (for comparison)")
    results = {}
    for gens in (20, 25):
        cfg = GAConfig(generations=gens, seed=seed)
        ga = GeneticAlgorithm(fitness_fn=ackley_vec, config=cfg)
        result = ga.run(verbose=False)
        print(
            f"  {gens:3d} generations -> best f = {result.best_fitness:.8f}, "
            f"x* = ({result.best_chromosome[0]:.4f}, {result.best_chromosome[1]:.4f})"
        )
        results[gens] = {
            "best_fitness": result.best_fitness,
            "best_chromosome": result.best_chromosome.tolist(),
        }
    return results


def main():
    result, cfg = run_main_ga()

    convergence_path = os.path.join(OUTPUT_DIR, "convergence_curve.png")
    surface_path = os.path.join(OUTPUT_DIR, "ackley_surface_solution.png")
    plot_convergence(result, convergence_path)
    plot_surface_with_solution(result, surface_path)
    print(f"\nSaved plots:\n  {convergence_path}\n  {surface_path}")

    classical = classical_comparison()
    reduced = reduced_generation_ga_runs()

    # Persist a summary of everything so it can be inspected / included in the report
    summary = {
        "ga_settings": {
            "population_size": cfg.pop_size,
            "generations": cfg.generations,
            "crossover_probability": cfg.pc,
            "mutation_probability": cfg.pm,
            "elitism": cfg.elitism,
            "selection": "Roulette Wheel",
            "crossover": "1-point",
            "mutation": "1-point Gaussian gene mutation",
            "bounds": cfg.bounds,
        },
        "ga_result": {
            "best_chromosome": result.best_chromosome.tolist(),
            "best_fitness": result.best_fitness,
            "generations_run": result.generations_run,
            "converged": result.converged,
        },
        "classical_comparison": classical,
        "reduced_generation_runs": reduced,
    }
    summary_path = os.path.join(OUTPUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved run summary: {summary_path}")


if __name__ == "__main__":
    main()

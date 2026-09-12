# GA-Assignment

Genetic Algorithm (GA) implementation that finds the **global minimum of the 2D Ackley function**, for the Machine Learning course assignment (Instructor: Roky Sir). See `Instructions/` for the original assignment brief.

## Problem

Minimize the 2D Ackley function:

$$f(x_1, x_2) = -a \cdot \exp\left(-b\sqrt{0.5\left(x_1^2+x_2^2\right)}\right) - \exp\left(0.5\left(\cos(cx_1)+\cos(cx_2)\right)\right) + a + \exp(1)$$

```py
f(x1, x2) = -a * exp(-b * sqrt(0.5 * (x1^2 + x2^2))) - exp(0.5 * (cos(c*x1) + cos(c*x2))) + a + exp(1)
```

with `a = 20`, `b = 0.2`, `c = 2π`, over the search domain `-5 <= x1, x2 <= 5`.
The known global minimum is `f(0, 0) = 0`.

## GA Design

| Setting             | Value                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| Chromosome          | `[x1, x2]` — real-value encoding, each gene in `[-5, 5]`                                        |
| Population size     | 50                                                                                              |
| Generations         | 100 (or early-stop once fitness ≤ 1e-6)                                                         |
| Crossover prob (Pc) | 80%                                                                                             |
| Mutation prob (Pm)  | 5%                                                                                              |
| Elitism             | Top 1 individual carried over unchanged                                                         |
| Selection           | Roulette Wheel (fitness-proportionate, inverted since we minimize)                              |
| Crossover           | 1-point crossover                                                                               |
| Mutation            | 1-point gene mutation — Gaussian perturbation of one randomly chosen gene, clipped to `[-5, 5]` |

## Project Structure

```
GA-Assignment/
├── README.md
├── docs/                          # Class slides (reference material)
├── Instructions/                  # Original assignment briefs and handwritten notes
└── src/                           # All code, inputs, and outputs live here
    ├── ackley.py                  # 2D Ackley function (scalar + vectorized)
    ├── genetic_algorithm.py       # GA engine: selection, crossover, mutation, elitism
    ├── main.py                    # Runs the GA, saves plots + a JSON run summary
    ├── test_ga.py                 # Unit tests (pytest)
    ├── requirements.txt
    └── output/                    # Generated on each run
        ├── convergence_curve.png
        ├── ackley_surface_solution.png
        └── run_summary.json
```

## How to Run

```bash
cd src
pip install -r requirements.txt
python3 main.py
```

This will:

1. Run the GA with the recommended settings above and print a generation-by-generation convergence log.
2. Save two plots to `src/output/`:
   - `convergence_curve.png` — best & mean fitness per generation
   - `ackley_surface_solution.png` — GA's best solution plotted on the Ackley surface, next to the true global minimum
    The convergence graph is also opened automatically in the system's default image viewer.
3. Run a short comparison against a classical optimizer (`scipy.optimize.minimize`, Nelder-Mead) starting from the same kind of point a GA individual might.
4. Run two reduced-generation GA runs (20 and 25 generations) for reference, as suggested in the assignment notes.
5. Save everything (settings, best solution, comparisons) to `src/output/run_summary.json`.

### Running the tests

```bash
cd src
python3 -m pytest test_ga.py -v
```

## Example Result

A typical run converges to a solution within ~0.02 of the true minimum well before the 100-generation cap, e.g.:

```
Best solution found: x1 = -0.001847, x2 = 0.004959
Best fitness f(x1, x2) = 0.01571263   (global minimum is 0 at (0,0))
```

## Understanding the GA Steps (for the exam / presentation)

1. **Initialization** — 50 random chromosomes `[x1, x2]`, each gene drawn uniformly from `[-5, 5]`.
2. **Fitness evaluation** — the Ackley function itself is the fitness function; lower values are fitter since this is a minimization problem.
3. **Elitism** — the single best individual each generation is copied unchanged into the next generation, guaranteeing the best-found solution is never lost.
4. **Selection (Roulette Wheel)** — since roulette wheel selection favors higher weights and we're minimizing, each individual's fitness is inverted (`worst_in_pop - fitness`) before computing selection probabilities, so the individual closest to 0 gets the largest wheel slice.
5. **Crossover (1-point)** — with probability 80%, two parents swap their `x2` genes around the single possible cut point in a 2-gene chromosome, producing two children.
6. **Mutation (1-point Gaussian)** — with probability 5%, one gene of a chromosome is perturbed by a small Gaussian-distributed random delta and clipped back into `[-5, 5]`.
7. **Replacement** — elite + offspring form the next generation of size 50.
8. **Stopping criteria** — 100 generations reached, or best fitness is close enough to 0 (`<= 1e-6`).

## Notes

- Keep any virtual environment folder out of git by adding it to `.gitignore`.

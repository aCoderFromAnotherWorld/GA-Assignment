# Genetic Algorithm (GA) Assignment — Ackley Function (2D) Global Minimum

**Course:** Machine Learning (ML) — Instructor: Roky Sir

## 1. Objective

Find the **global minimum of the 2D Ackley function** using a **Genetic Algorithm (GA)**.

## 2. Group & Submission Logistics

- Groups of **5 students**, **8 minutes** per group
- **24 groups total**, spread across **4 classes**
- Deadline: to be fixed by each group themselves
- Submission should include:
  - Full **code + output**
  - Be ready to explain your **understanding of GA** — this may also come up as an **exam question**, not just be graded as an assignment

## 3. The Ackley Function

**General n-dimensional form:**

```
f(x₁, x₂, …, xₙ) = −a·exp(−b·√((1/n)·Σxᵢ²)) − exp((1/n)·Σcos(c·xᵢ)) + a + exp(1)
```

**2D form (what you'll implement):**

```
f(x₁, x₂) = −a·exp(−b·√(0.5·(x₁² + x₂²))) − exp(0.5·(cos(c·x₁) + cos(c·x₂))) + a + exp(1)
```

**Recommended parameters:**

| Parameter | Value |
| --------- | ----- |
| a         | 20    |
| b         | 0.2   |
| c         | 2π    |

With these values, the function becomes:

```
f(x₁, x₂) = −20·exp(−0.2·√(0.5·(x₁² + x₂²))) − exp(0.5·(cos(2πx₁) + cos(2πx₂))) + 20 + exp(1)
```

**Search domain:** −5 ≤ x₁, x₂ ≤ 5

**Known global minimum:** f(x\*) = 0 at x\* = (0, 0) — i.e. f → 0 as x₁ → 0 and x₂ → 0

> A classical solver isn't needed — the GA should discover this minimum on its own. A result **close to zero** (the closer, the more accurate) counts as correct convergence.

## 4. Chromosome Encoding

- **Genes:** 2 → `[x₁, x₂]`
- **Encoding type:** Value (real-number) encoding
- Each gene initialized within **[-5, 5]**

## 5. GA Settings (recommended — adjustable)

| Setting                    | Value                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Population size            | 50                                                                                                                                                     |
| Number of generations      | 100 (or run until convergence) — used as stopping criterion                                                                                            |
| Crossover probability (Pc) | 80%                                                                                                                                                    |
| Mutation probability (Pm)  | 5%                                                                                                                                                     |
| Elitism                    | Yes — top 1 individual (best fitness) carried over unchanged each generation                                                                           |
| Selection method           | Roulette Wheel                                                                                                                                         |
| Crossover method           | 1-point crossover                                                                                                                                      |
| Mutation method            | 1-point gene mutation — pick one of the two genes and slightly perturb its value (e.g. a Gaussian-distributed random delta), keeping it within [-5, 5] |

## 6. Algorithm Outline

1. **Initialize** population: random chromosomes `[x₁, x₂]` within [-5, 5]
2. **Loop** for each generation:
   1. **Evaluate fitness** of every chromosome using the Ackley function (lower = fitter, since we're minimizing)
   2. **Sort** all individuals by fitness (ascending)
   3. **Elitism:** set aside the lowest-value (best) individual(s), unchanged
   4. **Selection:** pick parents via Roulette Wheel selection
   5. **Crossover:** apply 1-point crossover (probability 80%) to produce offspring
   6. **Mutation:** apply a small random perturbation to one gene (probability 5%)
   7. **Form the next generation:** combine elite individual(s) + new offspring; re-evaluate fitness and replace weaker individuals with better ones found
   8. **Check stopping criteria:** 100 generations reached, or fitness value close enough to 0
3. **Repeat** until stopping criteria is met

**Mutation example (illustrative, from notes):**
`[-1.5, -1.0]` → mutate one gene by a small delta → `[-1.0, -1.0]` (new solution)

## 7. Deliverables Checklist

- [ ] Working GA implementation (code)
- [ ] Output showing convergence toward `f(0, 0) ≈ 0`
- [ ] Clear understanding of each GA step (selection, crossover, mutation, elitism) — may be tested via exam questions

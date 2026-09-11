# GA Assignment: Finding Global Minimum of Ackley Function

**Instructor:** ML - Roky Sir  
**Group Size:** 5 members  
**Presentation Duration:** 8 minutes  
**Deadline:** To be fixed  
**Total Groups:** 24 groups across 4 classes

---

## 🎯 Assignment Objective

- Find the **global minimum of the Ackley function in 2D** using a Genetic Algorithm (GA).
- The Ackley function must be used as the **Fitness Function**.

---

## 🧮 Mathematical Formulation

### Ackley Function for nD

The general n-dimensional Ackley function is defined as:

$$f(x_1, x_2, \dots, x_n) = -a \cdot \exp\left(-b \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^n \cos(c x_i)\right) + a + \exp(1)$$

### Recommended Parameters

- $a = 20$
- $b = 0.2$
- $c = 2\pi$

### 2D Ackley Function Formulation

For 2D, the function expands to:

$$f(x_1, x_2) = -20 \exp\left(-0.2 \sqrt{0.5(x_1^2 + x_2^2)}\right) - \exp\left(0.5(\cos(2\pi x_1) + \cos(2\pi x_2))\right) + 20 + 2.71828$$

_(Note: $2.71828$ is the numerical value of $\exp(1)$)_

### Global Minimum & Range

- **Global Minimum:** $f(x^*) = 0$, where $x^* = 0$ (i.e., $x_1 \to 0, x_2 \to 0$).
- In the 2D case, $f(x_1, x_2) \approx f(0, 0) \approx 0$.
- **Search Range:** $-5 \le x_1, x_2 \le 5$

---

## ⚙️ GA Settings & Configuration

- **Population Size:** 50
- **Number of Generations:** 100 (Stopping Criteria) or infinite
- **Crossover Probability ($P_c$):** 80%
- **Mutation Probability ($P_m$):** 5%
- **Elitism:** 1 (Must be used)
- **Chromosome Representation:** $[x_1, x_2]$ using **Value Encoding** within the range $[-5, +5]$.
  - _Example Chromosome:_ $[-1.5, -1.0]$ or $[-4, 2.5]$

---

## 🧬 Genetic Operators Implementation

- **Initialization:** Population initialized with $x_1, x_2$. It does not strictly need a Gaussian distribution, just values roughly near the expected range.
- **Selection:** Roulette Wheel (Pie-chart method)
- **Crossover:** 1-point crossover
- **Mutation:** 1-point gene flip based on probability. For Value Encoding, this means slightly changing the value of one gene within the range.
  - _Example:_ $[-1.5, -1.0] \xrightarrow{+1} [-1.0, -1.0]$ (New solution)

---

## 🔄 Algorithm Flow / Steps

1. **Initialize Population:** Generate initial $x_1, x_2$ values.
2. **Evaluate Fitness:** Calculate fitness using the Ackley function.
3. **Sort:** Sort the population (Ascending/Descending) based on fitness values.
4. **Elitism:** Identify the elite solution (the one with the lowest value, i.e., the fittest) and keep it.
5. **Selection:** Select parents using Roulette Wheel.
6. **Crossover:** Apply 1-point crossover to generate offspring.
7. **Mutation:** Apply mutation (slight value change) based on probability.
8. **Replacement:** Calculate the fitness of the new offspring. If better than the elite, replace it. Create the next generation using the Best 50 individuals.
9. **Loop:** Repeat steps 2-8 until the stopping criteria (100 generations) is met.
10. **Convergence:** The algorithm should get closer to zero the more it runs (the closer to zero, the more accurate the result).

---

## 📝 Deliverables & Exam Notes

- **Assignment Submission:** Must include the Code, Output, and an explanation of how the GA was implemented.
- **Comparison:** Apply a classical algorithm and run GA for 20/25 iterations to compare and find the minimum.
- **Exam Prep:** Be prepared to answer questions about where GA is used and how it is implemented.

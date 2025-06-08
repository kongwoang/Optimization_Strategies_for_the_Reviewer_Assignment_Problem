# Optimization Strategies for the Reviewer Assignment Problem

This repository accompanies the technical report **“Optimization Strategies for the Reviewer Assignment Problem”**. It contains all source code, datasets, and experiment scripts referenced in the paper, plus instructions to reproduce every figure and table.

> **Goal**
> Assign a fixed number of reviewers to every paper while **minimising** the maximum load on any reviewer under eligibility constraints.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Description](#problem-description)
3. [Mathematical Model](#mathematical-model)
4. [Solution Approaches](#solution-approaches)

   * [Integer Linear Programming (ILP)](#integer-linear-programming-ilp)
   * [Greedy Algorithm](#greedy-algorithm)
   * [Hill‑Climbing Local Search (HCLS)](#hill-climbing-local-search-hcls)
   * [Adaptive Large Neighborhood Search (ALNS)](#adaptive-large-neighborhood-search-alns)
   * [Genetic Programming (GP)](#genetic-programming-gp)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Usage & Reproducibility](#usage--reproducibility)
8. [Conclusion](#conclusion)
9. [Acknowledgements](#acknowledgements)
10. [References](#references)

---

## Introduction

The **Reviewer Assignment Problem (RAP)** appears in academic conference management. Each submitted paper must receive *b* independent reviews from eligible reviewers, but no reviewer should be overloaded. RAP is **NP‑hard**, motivating both exact and heuristic methods.

This project implements and compares five optimisation strategies ranging from Integer Linear Programming to Genetic Programming, reporting their strengths, weaknesses, and empirical performance.

## Problem Description

* **Inputs**

* Set of papers $\mathcal{P}$ and reviewers $\mathcal{R}$.
* Eligibility list $L(i)$ for each paper $i$.
* Target number of reviews per paper $b$.

**Constraints**: Each paper must be assigned exactly $b$ eligible reviewers.  
**Objective**: Minimise the **maximum** number of papers assigned to any single reviewer.


## Mathematical Model

Let \$x\_{ij}=1\$ if reviewer \$j\$ is assigned to paper \$i\$ and 0 otherwise. Define the maximum load variable \$z\$.

```math
\min z = \max_{j \in \mathcal R} \sum_{i \in \mathcal P} x_{ij}
```

Subject to

* Paper coverage: \$\sum\_{j\in L(i)} x\_{ij} = b ;; \forall i \in \mathcal P\$
* Eligibility only: \$x\_{ij}=0\$ if \$j \notin L(i)\$
* Binary variables: \$x\_{ij}\in{0,1}\$

## Solution Approaches

### Integer Linear Programming (ILP)

* **Solvers used**  OR‑Tools (pywraplp & CP‑SAT) and Gurobi.
* **Pros** Optimal solutions on small/medium instances.
* **Cons** Runtime grows sharply with instance size.

### Greedy Algorithm

Assign reviewers iteratively, always picking the reviewer with (i) lowest current load and (ii) fewest remaining candidate papers.

* **Pros** Light‑weight, no parameter tuning, \$\mathcal O(N\log N)\$.
* **Cons** Sub‑optimal in about half the test cases.

### Hill‑Climbing Local Search (HCLS)

Start from a random feasible assignment and swap overloaded reviewers until no improving move exists.

* **Pros** Better quality than Greedy (optimal ≈ 58% of tests).
* **Cons** Can stall in local optima.

### Adaptive Large Neighborhood Search (ALNS)

Iteratively *destroy* and *repair* parts of the solution. Operator choice is learned online via scores.

* **Pros** Best workload **fairness**; escapes local optima.
* **Cons** Longer runtime; more parameters.

### Genetic Programming (GP)

Evolves assignment heuristics represented as expression trees composed of features such as load, slack, degree, etc.

* **Pros** Automatically discovers nonlinear policies.
* **Cons** Most computationally expensive; inconsistent on large data.

## Experiments

* **Dataset** 55 synthetic instances (50 – 20 000 papers) across Uniform, Gaussian, Poisson, Exponential and custom *Adversarial* distributions.
* **Platform** GitHub Codespaces ‑ 2‑core AMD EPYC 7763, 8 GB RAM, Ubuntu 20.04, Python 3.12.1.
* **Metrics** Maximum load, runtime, load variance.

## Results

| Method     | Optimal cases (55) | Avg. runtime          | Notes                  |
| ---------- | ------------------ | --------------------- | ---------------------- |
| **Gurobi** | 55                 | Fastest ILP           | Small/medium only      |
| **Greedy** | 25                 | ***Fastest overall*** | No fairness guarantee  |
| **HCLS**   | 32                 | Moderate              | Good quality vs. speed |
| **ALNS**   | 29                 | Slower                | Best fairness          |
| **GP**     | 9                  | Slowest               | High variance          |

Full tables & plots are in `/report/figures` and the LaTeX report.

## Usage & Reproducibility

1. **Clone repo**

   ```bash
   git clone https://github.com/kongwoang/Optimization_Strategies_for_the_Reviewer_Assignment_Problem.git
   cd Optimization_Strategies_for_the_Reviewer_Assignment_Problem
   ```


## Conclusion

* **ILP** (Gurobi) optimal but limited scalability.
* **Greedy** best when time is critical.
* **HCLS** strong quality–speed trade‑off.
* **ALNS** recommended when workload fairness matters.
* **GP** promising for automatic heuristic discovery; needs further optimisation.

Future work: hybrid methods, real conference data, explicit reviewer preferences / conflicts.

## Acknowledgements

We thank **Dr Pham Quang Dung** for guidance throughout the *Fundamentals of Optimization* course, and teammates **Le Tien Hop, Tran Phong Quan, Nguyen Gia Minh** for their contributions.

## References

Full bibliography is available in `report/references.bib` and the accompanying IEEE‑format PDF.

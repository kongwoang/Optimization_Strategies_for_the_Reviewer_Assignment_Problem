# Optimization Strategies for the Reviewer Assignment Problem

This project explores optimization strategies for the Reviewer Assignment Problem (RAP), a key challenge in academic conference management. The goal is to assign a fixed number of reviewers to each paper, minimizing the maximum load on any reviewer while respecting eligibility constraints.

## Table of Contents
- [Introduction](#introduction)
- [Problem Description](#problem-description)
- [Mathematical Model](#mathematical-model)
- [Solution Approaches](#solution-approaches)
  - [Integer Linear Programming (ILP)](#integer-linear-programming-ilp)
  - [Greedy Algorithm](#greedy-algorithm)
  - [Hill-Climbing Local Search](#hill-climbing-local-search)
  - [Adaptive Large Neighborhood Search (ALNS)](#adaptive-large-neighborhood-search-alns)
  - [Genetic Programming (GP)](#genetic-programming-gp)
- [Experiments](#experiments)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Introduction
The Reviewer Assignment Problem (RAP) involves assigning submitted papers to suitable reviewers, ensuring each paper receives fair and expert reviews while balancing reviewer workloads. As the number of papers and reviewers grows, the problem becomes computationally challenging (NP-hard).

This project presents and compares several methods for solving RAP, focusing on solution quality, computational efficiency, and workload balance.

## Problem Description
- **Inputs:** Sets of papers and reviewers, eligibility lists for each paper, and a fixed number of reviewers per paper.
- **Constraints:** Each paper must be assigned exactly _b_ reviewers from its eligible list.
- **Objective:** Minimize the maximum number of papers assigned to any reviewer (i.e., balance the load).

## Mathematical Model
Let:
- $\mathcal{P}$: Set of papers
- $\mathcal{R}$: Set of reviewers
- $L(i)$: Eligible reviewers for paper $i$
- $b$: Number of reviewers per paper
- $x_{ij}$: Binary variable (1 if reviewer $j$ is assigned to paper $i$)
- $z$: Maximum reviewer load

**Objective:**
$$
\min z = \max_{j \in \mathcal{R}} \sum_{i \in \mathcal{P}} x_{ij}
$$

**Constraints:**
- Each paper assigned to exactly $b$ eligible reviewers
- Reviewer assignments respect eligibility

## Solution Approaches
### Integer Linear Programming (ILP)
- **Solvers:** OR-Tools (pywraplp, CP-SAT), Gurobi
- **Strengths:** Finds optimal solutions for small/medium instances
- **Limitations:** Scalability issues for large datasets

### Greedy Algorithm
- Assigns reviewers with the lowest current load and fewest eligible papers to each paper.
- **Pros:** Fast, simple, no parameter tuning
- **Cons:** May not always find the optimal solution

### Hill-Climbing Local Search
- Starts from a random feasible solution and iteratively improves by swapping assignments to reduce the maximum load.
- **Pros:** Better solution quality than greedy, reasonable runtime
- **Cons:** Can get stuck in local optima

### Adaptive Large Neighborhood Search (ALNS)
- Alternates between destroying and repairing parts of the solution using adaptive heuristics.
- **Pros:** Balances load and fairness, escapes local optima
- **Cons:** Slower than greedy/HCLS, more complex

### Genetic Programming (GP)
- Evolves assignment policies as tree-based programs using genetic operators.
- **Pros:** Learns complex heuristics automatically
- **Cons:** High computational cost, less consistent performance

## Experiments
- **Dataset:** 55 synthetic test cases with varying numbers of papers, reviewers, and eligibility distributions (Uniform, Gaussian, Poisson, Exponential, Adversarial).
- **Environment:** GitHub Codespaces (2-core AMD EPYC, 8GB RAM, Ubuntu 20.04, Python 3.12.1)
- **Metrics:** Solution quality (max reviewer load), runtime, fairness (variance in load)

## Results
- **ILP (Gurobi):** Best for small/medium instances, optimal solutions, fastest among ILP solvers.
- **Greedy:** Fastest overall, optimal in ~25/55 cases.
- **HCLS:** Optimal in ~32/55 cases, better than greedy for most others.
- **ALNS:** Slightly slower, but best for fairness and balancing workloads.
- **GP:** Most flexible, but slowest and less consistent.

See the report for detailed tables and plots.

## Conclusion
- **ILP** is best for small instances.
- **Greedy** is suitable for quick, large-scale assignments.
- **HCLS** offers a good balance of speed and quality.
- **ALNS** is recommended when fairness is critical.
- **GP** shows promise for learning assignment policies but needs further optimization.

Future work includes hybrid methods, real-world datasets, and support for reviewer preferences/conflicts.

## Acknowledgements
Special thanks to Dr. Pham Quang Dung for guidance and support, and to all team members for their collaboration and technical contributions.

## References
- [Project Repository](https://github.com/kongwoang/Optimization_Strategies_for_the_Reviewer_Assignment_Problem)
- See the report for full bibliography.

---

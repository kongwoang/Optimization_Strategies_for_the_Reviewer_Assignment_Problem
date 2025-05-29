import random
import numpy as np
import matplotlib.pyplot as plt
import os

def write_L_to_file(L, N, M, b, distribution_name, output_dir="/content/sample_data"):
    filename = f"{distribution_name}_{N}_{M}_{b}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"{N} {M} {b}\n")
        for i in range(1, N + 1):
            reviewers = L[i]
            f.write(f"{len(reviewers)} {' '.join(map(str, reviewers))}\n")

    return filepath
# ---------- Data Generators (paper → reviewers) ----------

def gen_L_uniform(N, M, min_k, max_k):
    ks = np.random.randint(min_k, max_k + 1, size=N)
    return {i+1: random.sample(range(1, M+1), ks[i]) for i in range(N)}

def gen_L_gaussian(N, M, mean, std, min_k, max_k):
    ks = []
    while len(ks) < N:
        s = int(round(random.gauss(mean, std)))
        if min_k <= s <= max_k:
            ks.append(s)
    return {i+1: random.sample(range(1, M+1), ks[i]) for i in range(N)}

def gen_L_poisson(N, M, lam, min_k, max_k):
    """
    k ~ Poisson(lam), rejection‐sampled into [min_k..max_k]
    """
    ks = []
    while len(ks) < N:
        s = np.random.poisson(lam)
        if min_k <= s <= max_k:
            ks.append(s)
    return {i+1: random.sample(range(1, M+1), ks[i]) for i in range(N)}

def gen_L_exponential(N, M, scale, min_k, max_k):
    """
    k ~ Exp(scale), rejection‐sampled into [min_k..max_k]
    """
    ks = []
    while len(ks) < N:
        s = int(round(np.random.exponential(scale)))
        if min_k <= s <= max_k:
            ks.append(s)
    return {i+1: random.sample(range(1, M+1), ks[i]) for i in range(N)}

def gen_bait_and_trap(N, M, b, rare_group_size=5, rare_per_paper=1, common_pool_size=None, burn_rate=0.7):
    """
    - rare_group_size: number of 'rare' reviewers.
    - rare_per_paper: how many rare reviewers each paper initially shows.
    - common_pool_size: number of 'common' reviewers (default = M - rare_group_size)
    
    Papers 1..P0: only rare reviewers + common ones
    Papers P0+1..N: only common reviewers
    Greedy will use up rares on the first block, then be forced to use common 
    ones exclusively on the second block—but common ones get overloaded late.
    """
    if common_pool_size is None:
        common_pool_size = M - rare_group_size

    rare = list(range(1, rare_group_size + 1))
    common = list(range(rare_group_size + 1, rare_group_size + common_pool_size + 1))
    L = {}

    # Divide papers into two blocks:
    #   Block A: first 70% of papers (to bait greedy)
    #   Block B: last 30% of papers (trap)
    P0 = int(burn_rate * N)

    for i in range(1, N + 1):
        if i <= P0:
            # Block A: give each paper a few rare reviewers + a few common
            reviewers = []
            # ensure at least one rare is available (greedy will pick them)
            reviewers += random.sample(rare, rare_per_paper)
            # fill up to, say, 2*b slots with random common
            reviewers += random.sample(common, b * 2 - rare_per_paper)
        else:
            # Block B: no rares here—only common reviewers
            reviewers = random.sample(common, b * 2)
        L[i] = reviewers

    return L


# ---------- Scenario Setup & Plotting ----------
"""
        (50, 20,2),
        (100, 50 , 3),
        (200, 100, 3),
        (500, 350, 4),
        (800, 500, 5),
        (1000, 700, 5),
        (2000, 900, 5),
        (5000, 2000, 6),
        (10000, 4000, 6),
        (20000, 9000, 6)
"""
N, M, b = 20000, 9000, 6
min_u, max_u = 2, min(20,M//4)
min_g, max_g = 2, min(10,M//4)
min_p, max_p = 2, min(20,M//3)
min_e, max_e = 2, min(30,M//2)
mean, std = (min_g+max_g)//2, 2            # widen Gaussian a bit (min_g < mean < max_g, mean ~ (min+max)/2, std should remain still)
lam = 5                       # Poisson λ (min_p < lam < max_p, 2 situation: lam near min_p / near max_p)
scale = 3                     # Exponential scale (min_e < scale < max_e, 2 situation: lam near min_e / near max_e)
rare_group=int(0.15*M)
rare_freq=1
burn_rate=0.6
scenarios = {
    "Uniform":   gen_L_uniform(N, M, b+min_u, b+max_u),
    "Gaussian":  gen_L_gaussian(N, M, b+mean, std, b+min_g, b+max_g),
    "Poisson":   gen_L_poisson(N, M, b+lam, b+min_p, b+max_p),
    "Exponential": gen_L_exponential(N, M, b+scale, b+min_e, b+max_e),
    "Adversarial": gen_bait_and_trap(N,M,b, rare_group_size=rare_group, rare_per_paper=rare_freq, burn_rate=burn_rate),
}


output_dir="E:\\20242\\Fundamentals of Optimization\\own_prj\\Optimization_Strategies_for_the_Reviewer_Assignment_Problem\\datasets"
# Generate files for each distribution
generated_files = {}
for name, L in scenarios.items():
    path = write_L_to_file(L, N, M, b, name, output_dir= output_dir)
    generated_files[name] = path
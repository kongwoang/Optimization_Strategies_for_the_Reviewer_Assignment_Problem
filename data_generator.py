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

def gen_L_adversarial(N, M, b, groups, overlap):
    L = {}
    per_group = N // groups
    for g in range(groups):
        popular = random.sample(range(1, M+1), max(1, int(overlap*M)))
        for i in range(g*per_group+1, (g+1)*per_group+1):
            num_pop = min(len(popular), b-1)
            chosen = random.sample(popular, num_pop)
            rest   = random.sample([r for r in range(1, M+1) if r not in chosen], b-num_pop)
            L[i] = chosen + rest
    for i in range(groups*per_group+1, N+1):
        L[i] = random.sample(range(1, M+1), b)
    return L

# ---------- Params Tuning ----------

N, M, b = 2000, 100, 10 
min_u, max_u = 3, 10
min_g, max_g = 3, 10
min_p, max_p = 3, 20
min_e, max_e = 2, 30
mean, std = 6, 1.5            # widen Gaussian a bit (min_g < mean < max_g, mean ~ (min+max)/2, std should remain still)
lam = 16                       # Poisson λ (min_p < lam < max_p, 2 situation: lam near min_p / near max_p)
scale = 3                     # Exponential scale (min_e < scale < max_e, 2 situation: lam near min_e / near max_e)

scenarios = {
    "Uniform":   gen_L_uniform(N, M, b+min_u, b+max_u),
    "Gaussian":  gen_L_gaussian(N, M, b+mean, std, b+min_g, b+max_g),
    "Poisson":   gen_L_poisson(N, M, b+lam, b+min_p, b+max_p),
    "Exponential": gen_L_exponential(N, M, b+scale, b+min_e, b+max_e),
    "Adversarial": gen_L_adversarial(N, M, b, groups=4, overlap=0.2),
}

current_dir = os.getcwd()
output_dir=os.path.join(current_dir, 'datasets')
# Generate files for each distribution
generated_files = {}
for name, L in scenarios.items():
    path = write_L_to_file(L, N, M, b, name, output_dir= output_dir)
    generated_files[name] = path
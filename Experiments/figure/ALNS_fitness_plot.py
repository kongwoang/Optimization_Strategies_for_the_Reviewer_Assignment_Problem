import os, time, random
import matplotlib.pyplot as plt

# ---------- các hàm tiện ích gốc ----------------
def variance(values):
    n = len(values)
    if n == 0:
        return 0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n

def add_assignment(sol, loads, paper_idx, reviewers):
    sol[paper_idx] = reviewers[:]
    for r in reviewers:
        loads[r] += 1

def remove_assignment(sol, loads, paper_idx):
    for r in sol[paper_idx]:
        loads[r] -= 1
    sol[paper_idx].clear()

def fitness(loads, M, avg_load, alpha=0.9, beta=0.05, gamma=0.05):
    load_values = [loads.get(j, 0) for j in range(1, M + 1)]
    maxL = max(load_values)
    varL = variance(load_values)
    overload_count = sum(1 for l in load_values if l > avg_load)
    return alpha * maxL + beta * varL + gamma * overload_count

def initial_solution(N, M, b, L):
    sol   = [[] for _ in range(N)]
    loads = {i: 0 for i in range(1, M + 1)}
    for i in range(N):
        cand = sorted(L[i], key=lambda r: loads[r])
        add_assignment(sol, loads, i, cand[:b])
    return sol, loads

def random_destroy(sol, loads, ratio):
    N = len(sol)
    k = max(1, int(N * ratio))
    removed = random.sample(range(N), k)
    for i in removed:
        remove_assignment(sol, loads, i)
    return removed

def worst_load_destroy(sol, loads, ratio):
    N = len(sol)
    k = max(1, int(N * ratio))
    idx = sorted(range(N),
                 key=lambda i: sum(loads.get(r, 0) for r in sol[i]),
                 reverse=True)[:k]
    for i in idx:
        remove_assignment(sol, loads, i)
    return idx

def greedy_repair(sol, loads, removed, b, L):
    for i in removed:
        cand = sorted(L[i], key=lambda r: loads[r])
        add_assignment(sol, loads, i, cand[:b])

def random_repair(sol, loads, removed, b, L):
    for i in removed:
        cand = L[i][:]
        random.shuffle(cand)
        add_assignment(sol, loads, i, cand[:b])

def choose(ops, weights):
    return random.choices(range(len(ops)), weights=weights, k=1)[0]

def update_weights(weights, idx, reward, decay=0.9):
    for i in range(len(weights)):
        if i == idx:
            weights[i] = decay * weights[i] + reward
        else:
            weights[i] = decay * weights[i]

# ---------- ALNS core ---------------------------------------
def alns(N, M, b, L, max_iter=1000, seed=42, track_history=False):
    random.seed(seed)
    destroy_ops = [random_destroy, worst_load_destroy]
    repair_ops  = [greedy_repair, random_repair]
    dw, rw = [1.0]*len(destroy_ops), [1.0]*len(repair_ops)

    avg_load        = b * N / M
    current, loads  = initial_solution(N, M, b, L)
    best_val        = fitness(loads, M, avg_load)

    history = [best_val] if track_history else None

    for _ in range(max_iter):
        didx = choose(destroy_ops, dw)
        ridx = choose(repair_ops,  rw)

        saved_sol  = [r[:] for r in current]
        saved_load = loads.copy()

        removed = destroy_ops[didx](current, loads, ratio=0.15)
        repair_ops[ridx](current, loads, removed, b, L)
        val = fitness(loads, M, avg_load)

        if val < best_val:
            best_val = val
            update_weights(dw, didx, reward=2)
            update_weights(rw, ridx, reward=2)
        else:
            current, loads = saved_sol, saved_load
            update_weights(dw, didx, reward=0.1)
            update_weights(rw, ridx, reward=0.1)

        if track_history:
            history.append(best_val)

    return (loads, history) if track_history else loads

# ---------- I/O ---------------------------------------------
def read_instance(path):
    with open(path, "r") as f:
        N, M, b = map(int, f.readline().split())
        L = []
        for _ in range(N):
            tokens = list(map(int, f.readline().split()))
            k, reviewers = tokens[0], tokens[1:]
            assert len(reviewers) == k
            L.append(reviewers)
    return N, M, b, L

# ---------- MAIN + Plotting ----------------------------------
if __name__ == "__main__":
    path = "instances\\Adversarial_800_50_5.txt"  # ⚠️ chỉnh lại nếu cần
    N, M, b, L = read_instance(path)

    _, history = alns(N, M, b, L, max_iter=1000, seed=42, track_history=True)

    plt.figure(figsize=(5, 4))
    plt.plot(history, marker='o', markersize=3, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    # plt.title("ALNS Fitness Value over Iterations")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Experiments\\figure\\ALNS_fitness_plot.png', dpi=300, bbox_inches='tight')

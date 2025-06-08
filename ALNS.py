import os, time, random

# ---------- các hàm tiện ích gốc (giữ nguyên logic) ----------------
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

def alns(N, M, b, L, max_iter=1000, seed=42):
    random.seed(seed)
    destroy_ops = [random_destroy, worst_load_destroy]
    repair_ops  = [greedy_repair, random_repair]
    dw, rw = [1.0]*len(destroy_ops), [1.0]*len(repair_ops)

    avg_load        = b * N / M
    current, loads  = initial_solution(N, M, b, L)
    best_val        = fitness(loads, M, avg_load)

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

    return loads            # trả về dict tải reviewer

# ---------- I/O ----------------------------------------------------
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

def write_result(out_path, n, m, obj, runtime_ms):
    with open(out_path, "w") as f:
        f.write(f"{os.path.basename(out_path)}\n")
        f.write(f"n = {n}\n")
        f.write(f"m = {m}\n")
        f.write(f"Objective Value: {obj}\n")
        f.write(f"{runtime_ms} ms\n")

# ---------- batch runner -------------------------------------------
def main():
    root = os.getcwd()
    inst_dir = os.path.join(root, "instances")
    res_dir  = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    files = [f for f in os.listdir(inst_dir) if f.endswith(".txt")]

    for fname in files:
        in_path  = os.path.join(inst_dir, fname)
        out_name = f"[ALNS] {fname}"
        out_path = os.path.join(res_dir, out_name)

        print(f"Đang xử lý: {fname} → {out_name}")
        N, M, b, L = read_instance(in_path)

        start = time.time()
        loads = alns(N, M, b, L, max_iter=1000, seed=42)
        runtime = int((time.time() - start) * 1000)

        max_load = max(loads.values()) if loads else 0
        write_result(out_path, N, M, max_load, runtime)

if __name__ == "__main__":
    main()

import sys, random

# ------------------------- đọc dữ liệu -------------------------
def read_instance():
    N, M, b = map(int, sys.stdin.readline().split())
    L = []
    for _ in range(N):
        tokens = list(map(int, sys.stdin.readline().split()))
        k, reviewers = tokens[0], tokens[1:]
        assert len(reviewers) == k
        L.append(reviewers)
    return N, M, b, L

# ------------------ tính variance thủ công ---------------------
def variance(values):
    n = len(values)
    if n == 0:
        return 0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n

# ----------------- các hàm tiện ích cho nghiệm -----------------
def add_assignment(sol, loads, paper_idx, reviewers):
    sol[paper_idx] = reviewers[:]
    for r in reviewers:
        loads[r] += 1

def remove_assignment(sol, loads, paper_idx):
    for r in sol[paper_idx]:
        loads[r] -= 1
    sol[paper_idx].clear()

# --------------------- Hàm fitness đa yếu tố -------------------
def fitness(loads, M, avg_load, alpha=1.0, beta=0.5, gamma=0.2):
    load_values = [loads.get(j, 0) for j in range(1, M + 1)]
    maxL = max(load_values)
    varL = variance(load_values)
    overload_count = sum(1 for l in load_values if l > avg_load)
    return alpha * maxL + beta * varL + gamma * overload_count

# -------------- khởi tạo – greedy tải thấp nhất -----------------
def initial_solution(N, M, b, L):
    sol = [[] for _ in range(N)]
    loads = {i: 0 for i in range(1, M + 1)}
    for i in range(N):
        cand = sorted(L[i], key=lambda r: loads[r])
        add_assignment(sol, loads, i, cand[:b])
    return sol, loads

# -------------- Các operator phá huỷ / xây lại ------------------
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

# ------------- Chọn operator bằng roulette-wheel ----------------
def choose(ops, weights):
    return random.choices(range(len(ops)), weights=weights, k=1)[0]

# ------------- Cập nhật trọng số thích nghi ---------------------
def update_weights(weights, idx, reward, decay=0.9):
    for i in range(len(weights)):
        if i == idx:
            weights[i] = decay * weights[i] + reward
        else:
            weights[i] = decay * weights[i]

# ------------- Thuật toán ALNS không dùng SA -------------------
def alns(N, M, b, L, max_iter=1000, seed=0):
    random.seed(seed)
    destroy_ops = [random_destroy, worst_load_destroy]
    repair_ops  = [greedy_repair, random_repair]
    dw, rw = [1.0]*len(destroy_ops), [1.0]*len(repair_ops)

    avg_load = b * N / M
    current, loads = initial_solution(N, M, b, L)
    best     = [r[:] for r in current]
    best_val = fitness(loads, M, avg_load)

    for _ in range(max_iter):
        # --- chọn operator ---
        didx = choose(destroy_ops, dw)
        ridx = choose(repair_ops,  rw)

        # --- sao lưu ---
        saved_sol  = [r[:] for r in current]
        saved_load = loads.copy()

        # --- áp dụng ---
        removed = destroy_ops[didx](current, loads, ratio=0.15)
        repair_ops[ridx](current, loads, removed, b, L)
        val = fitness(loads, M, avg_load)

        # --- chỉ nhận nếu tốt hơn ---
        if val < best_val:
            best_val = val
            best = [r[:] for r in current]
            update_weights(dw, didx, reward=2)
            update_weights(rw, ridx, reward=2)
        else:
            current, loads = saved_sol, saved_load
            update_weights(dw, didx, reward=0.1)
            update_weights(rw, ridx, reward=0.1)

    return best, loads

# --------------------------- main -------------------------------
if __name__ == "__main__":
    N, M, b, L = read_instance()
    solution, loads = alns(N, M, b, L, max_iter=1000, seed=42)

    print(N)
    for reviewers in solution:
        print(b, *reviewers)

    load_values = [loads[r] for r in sorted(loads)]
    avg = b * N / M
    print("Objective (max load):", max(load_values))
    print("Variance:", round(variance(load_values), 2))
    print("Overloaded reviewers:", sum(1 for l in load_values if l > avg))

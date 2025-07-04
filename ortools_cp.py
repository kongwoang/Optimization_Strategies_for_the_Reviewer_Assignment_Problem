import sys, random
from collections import Counter

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

# ----------------- các hàm tiện ích cho nghiệm -----------------
def max_load(loads):
    return max(loads.values()) if loads else 0

def add_assignment(sol, loads, paper_idx, reviewers):
    sol[paper_idx] = reviewers[:]
    for r in reviewers:
        loads[r] += 1

def remove_assignment(sol, loads, paper_idx):
    for r in sol[paper_idx]:
        loads[r] -= 1
    sol[paper_idx].clear()

# -------------- khởi tạo – greedy tải thấp nhất -----------------
def initial_solution(N, b, L):
    sol = [[] for _ in range(N)]
    loads = Counter()
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
                 key=lambda i: sum(loads[r] for r in sol[i]),
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

    current, loads = initial_solution(N, b, L)
    best     = [r[:] for r in current]
    best_val = max_load(loads)

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
        val = max_load(loads)

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

    return best

# --------------------------- main -------------------------------
if __name__ == "__main__":
    N, M, b, L = read_instance()
    solution = alns(N, M, b, L, max_iter=1000, seed=42)

    print(N)
    for reviewers in solution:
        print(b, *reviewers)

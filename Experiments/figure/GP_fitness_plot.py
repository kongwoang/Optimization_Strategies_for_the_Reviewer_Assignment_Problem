import random, copy, math, time, os
from typing import List, Tuple, Any
import matplotlib.pyplot as plt

# ----------------- 1. Đặt đường dẫn input tại đây -----------------
FILE_PATH = "instances\\Adversarial_200_15_3.txt"

# ----------------- 2. GP Config -----------------
MAX_DEPTH = 4
OP_NAMES = ["add", "sub", "mul", "safeDiv", "min", "max"]
TERM_NAMES = ["load", "slack", "deg", "candCnt", "rand", "const"]
OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "safeDiv": lambda a, b: a / b if abs(b) > 1e-9 else a,
    "min": min,
    "max": max,
}


# ----------------- 3. Đọc dữ liệu -----------------
def load_instance(path: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        N, M, B = map(int, f.readline().split())
        L = []
        for _ in range(N):
            nums = list(map(int, f.readline().split()))
            k, revs = nums[0], nums[1:]
            if len(revs) != k:
                raise ValueError("Sai định dạng: k ≠ số reviewer")
            L.append([r - 1 for r in revs])  # 0-based
    return N, M, B, L


# ----------------- 4. Bộ máy GP -----------------
def run_gp(N: int, M: int, B: int, L: List[List[int]], *,
           pop_size=200, max_generations=400,
           seed=42) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:

    random.seed(seed)
    start_time = time.time()

    global GLOBAL_QUOTA, REVIEWER_DEG
    GLOBAL_QUOTA = math.ceil(N * B / M)
    REVIEWER_DEG = [0] * M
    for i in range(N):
        for r in L[i]:
            REVIEWER_DEG[r] += 1

    def rev_idx(x): return int(abs(x)) % M
    def pap_idx(x): return int(abs(x)) % N
    def rand_const(): return random.uniform(-2, 2)

    def eval_tree(node: Any, r: int, i: int, loads: List[int]):
        t = node[0]
        if t == 'op':
            _, op, lft, rgt = node
            return OPS[op](eval_tree(lft, r, i, loads),
                           eval_tree(rgt, r, i, loads))
        _, name, val = node
        if name == 'load': return loads[rev_idx(r)]
        if name == 'slack': return GLOBAL_QUOTA - loads[rev_idx(r)]
        if name == 'deg': return REVIEWER_DEG[rev_idx(r)]
        if name == 'candCnt': return len(L[pap_idx(i)])
        if name == 'rand': return random.random()
        if name == 'const': return val
        raise ValueError

    def gen_full_tree(d):
        if d == 0:
            t = random.choice(TERM_NAMES)
            return ('term', t, rand_const() if t == 'const' else None)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_full_tree(d - 1), gen_full_tree(d - 1))

    def gen_grow_tree(d):
        if d == 0 or (d > 0 and random.random() < .3):
            t = random.choice(TERM_NAMES)
            return ('term', t, rand_const() if t == 'const' else None)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_grow_tree(d - 1), gen_grow_tree(d - 1))

    def mutate(t):
        if random.random() < .1 or t[0] == 'term':
            return gen_grow_tree(MAX_DEPTH)
        _, name, l, r = t
        if random.random() < .5:
            return ('op', name, mutate(l), r)
        return ('op', name, l, mutate(r))

    def crossover(a, b):
        if a[0] == 'term' or b[0] == 'term':
            return copy.deepcopy(b)
        if random.random() < .5:
            return ('op', a[1], crossover(a[2], b[2]), a[3])
        return ('op', a[1], a[2], crossover(a[3], b[3]))

    def build_assignment(tree):
        loads = [0] * M
        viol = 0
        sol = []
        for i in range(N):
            chosen = []
            for _ in range(B):
                cand = [r for r in L[i] if r not in chosen]
                if not cand:
                    viol += 1
                    break
                _, best = min((eval_tree(tree, r, i, loads), r) for r in cand)
                chosen.append(best)
                loads[best] += 1
            sol.append(chosen)
        return max(loads), viol, sol

    # init population
    depths = range(2, MAX_DEPTH + 1)
    per_d = pop_size // (2 * len(depths)) or 1
    pop = []
    for d in depths:
        pop.extend(gen_full_tree(d) for _ in range(per_d))
        pop.extend(gen_grow_tree(d) for _ in range(per_d))
    while len(pop) < pop_size:
        pop.append(gen_grow_tree(random.choice(tuple(depths))))

    fit_sol = [build_assignment(t) for t in pop]
    fit = [f[:2] for f in fit_sol]
    sols = [f[2] for f in fit_sol]

    best_idx = min(range(pop_size), key=lambda i: fit[i])
    best = (copy.deepcopy(pop[best_idx]), fit[best_idx], sols[best_idx])
    history = [fit[best_idx]]

    TOUR, CXPB, MUTPB = 5, .9, .1

    # evolution
    for gen in range(1, max_generations + 1):
        def select():
            idxs = random.sample(range(pop_size), TOUR)
            j = min(idxs, key=lambda k: fit[k])
            return copy.deepcopy(pop[j])

        new_pop = []
        for _ in range(pop_size):
            p1, p2 = select(), select()
            child = crossover(p1, p2) if random.random() < CXPB else copy.deepcopy(p1)
            if random.random() < MUTPB:
                child = mutate(child)
            new_pop.append(child)

        pop = new_pop
        fit_sol = [build_assignment(t) for t in pop]
        fit = [f[:2] for f in fit_sol]
        sols = [f[2] for f in fit_sol]

        cur_idx = min(range(pop_size), key=lambda i: fit[i])
        if fit[cur_idx] < best[1]:
            best = (copy.deepcopy(pop[cur_idx]), fit[cur_idx], sols[cur_idx])
        history.append(best[1])

    return best[1], history, best[2]


# ----------------- 5. Vẽ biểu đồ -----------------
def plot_history(history: List[Tuple[int, int]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    loads, viols = zip(*history)
    iters = range(len(loads))
    plt.figure()
    plt.plot(iters, loads, label="Max load")
    plt.plot(iters, viols, label="Violations")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("GP Fitness over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ----------------- 6. Run main -----------------
if __name__ == "__main__":
    N, M, B, L = load_instance(FILE_PATH)
    best_fit, hist, solution = run_gp(N, M, B, L, pop_size=200, max_generations=400, seed=0)

    print("Best result (max_load, violations):", best_fit)
    for i, revs in enumerate(solution, 1):
        print(f"Paper {i}: {revs}")

    plot_history(hist, "Experiments\\figure\\GP_fit_plot.png")
    print("Đã lưu biểu đồ tại → Experiments/figure/GP_fit_plot.png")

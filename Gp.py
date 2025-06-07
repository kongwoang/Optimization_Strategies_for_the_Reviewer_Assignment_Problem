
from __future__ import annotations
import os, time, copy, math, random
from typing import Any, List, Tuple

# ----------------------------------------------------------------
#  ── Instance I/O ­──
# ----------------------------------------------------------------
def read_instance(path: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        tok = [int(x) for line in f for x in line.strip().split()]
    N, M, b = tok[:3]
    idx = 3
    L: List[List[int]] = []
    for _ in range(N):
        k = tok[idx]; idx += 1
        revs = tok[idx:idx + k]; idx += k
        L.append([r - 1 for r in revs])           # 0-based
    return N, M, b, L

# ----------------------------------------------------------------
#  ── GP core (nguyên gốc, chỉ chuyển thành hàm) ──
# ----------------------------------------------------------------
OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b if abs(b) > 1e-9 else a,
    'min': lambda a, b: a if a < b else b,
    'max': lambda a, b: a if a > b else b,
}
OP_NAMES = list(OPS.keys())
TERM_NAMES = ['load', 'slack', 'deg', 'candCnt', 'rand', 'const']
MAX_DEPTH = 4

def run_gp(N: int, M: int, B: int, L: List[List[int]],
           pop_size=200, generations=40, seed=42, bad_init=False) -> Tuple[int, int]:
    random.seed(seed)

    global GLOBAL_QUOTA, REVIEWER_DEG
    GLOBAL_QUOTA = math.ceil(N * B / M)
    REVIEWER_DEG = [0] * M
    for i in range(N):
        for r in L[i]:
            REVIEWER_DEG[r] += 1

    # ───── helper fns ──────────────────────────
    def rev_idx(x): return int(abs(x)) % M
    def pap_idx(x): return int(abs(x)) % N
    def rand_const(): return random.uniform(-2, 2)

    def eval_tree(node: Any, r: int, i: int, loads: List[int]):
        t = node[0]
        if t == 'op':
            _, name, lft, rgt = node
            return OPS[name](eval_tree(lft, r, i, loads), eval_tree(rgt, r, i, loads))
        _, name, val = node
        if name == 'load':    return loads[rev_idx(r)]
        if name == 'slack':   return GLOBAL_QUOTA - loads[rev_idx(r)]
        if name == 'deg':     return REVIEWER_DEG[rev_idx(r)]
        if name == 'candCnt': return len(L[pap_idx(i)])
        if name == 'rand':    return random.random()
        if name == 'const':   return val
        raise ValueError

    # ── tree generation/mutation/crossover ──
    def gen_full_tree(depth):
        if depth == 0:
            term = random.choice(TERM_NAMES)
            val = rand_const() if term == 'const' else None
            return ('term', term, val)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_full_tree(depth - 1), gen_full_tree(depth - 1))

    def gen_grow_tree(depth):
        if depth == 0 or (depth > 0 and random.random() < 0.3):
            term = random.choice(TERM_NAMES)
            val = rand_const() if term == 'const' else None
            return ('term', term, val)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_grow_tree(depth - 1), gen_grow_tree(depth - 1))

    def gen_bad_tree():
        return ('term', 'const', random.uniform(50, 100))

    def mutate(tree):
        if random.random() < 0.1 or tree[0] == 'term':
            return gen_grow_tree(MAX_DEPTH)
        _, name, l, r = tree
        if random.random() < 0.5:
            return ('op', name, mutate(l), r)
        return ('op', name, l, mutate(r))

    def crossover(a, b):
        if a[0] == 'term' or b[0] == 'term':
            return copy.deepcopy(b)
        if random.random() < 0.5:
            return ('op', a[1], crossover(a[2], b[2]), a[3])
        return ('op', a[1], a[2], crossover(a[3], b[3]))

    # ── evaluation ──
    def build_assignment(f):
        loads = [0] * M
        viol = 0
        for i in range(N):
            assigned = []
            for _ in range(B):
                cands = [r for r in L[i] if r not in assigned]
                if not cands:
                    viol += 1; break
                scores = [(eval_tree(f, r, i, loads), r) for r in cands]
                _, best = min(scores, key=lambda t: t[0])
                assigned.append(best)
                loads[best] += 1
        return max(loads), viol

    # ───── GP main loop ─────
    if bad_init:
        pop = [gen_bad_tree() for _ in range(pop_size)]
    else:
        depths = list(range(2, MAX_DEPTH + 1))
        cnt = pop_size // (2 * len(depths))
        pop = []
        for d in depths:
            for _ in range(cnt):
                pop.append(gen_full_tree(d))
            for _ in range(cnt):
                pop.append(gen_grow_tree(d))
        while len(pop) < pop_size:
            d = random.choice(depths)
            pop.append(gen_grow_tree(d))

    fit = [build_assignment(t) for t in pop]
    best_idx = min(range(pop_size), key=lambda j: fit[j])
    best_fit = fit[best_idx]
    best_tree = copy.deepcopy(pop[best_idx])

    TOUR = 5; CXPB = 0.9; MUTPB = 0.1
    for _ in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            def select():
                ks = random.sample(range(pop_size), TOUR)
                ks.sort(key=lambda j: fit[j])
                return copy.deepcopy(pop[ks[0]])
            p1, p2 = select(), select()
            child = crossover(p1, p2) if random.random() < CXPB else copy.deepcopy(p1)
            if random.random() < MUTPB:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        fit = [build_assignment(t) for t in pop]
        idx = min(range(pop_size), key=lambda j: fit[j])
        if fit[idx] < best_fit:
            best_fit = fit[idx]; best_tree = copy.deepcopy(pop[idx])

    # best_fit = (max_load, violations)
    return best_fit  # tuple

# ----------------------------------------------------------------
#  ── Batch runner ­──
# ----------------------------------------------------------------
def write_result(path, n, m, obj, runtime_ms):
    with open(path, "w") as f:
        f.write(f"{os.path.basename(path)}\n")
        f.write(f"m = {m}\n")
        f.write(f"Objective Value: {obj}\n")
        f.write(f"{runtime_ms} ms\n")

def main():
    root = os.getcwd()
    inst_dir = os.path.join(root, "instances")
    res_dir  = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    files = [f for f in os.listdir(inst_dir) if f.endswith(".txt")]

    for fname in files:
        in_path  = os.path.join(inst_dir, fname)
        out_name = f"[GP] {fname}"
        out_path = os.path.join(res_dir, out_name)

        print(f"Đang xử lý: {fname} → {out_name}")
        N, M, B, L = read_instance(in_path)

        start = time.time()
        max_load, violations = run_gp(N, M, B, L, seed=42, bad_init=False)
        runtime = int((time.time() - start) * 1000)

        write_result(out_path, N, M, max_load, runtime)

if __name__ == "__main__":
    main()

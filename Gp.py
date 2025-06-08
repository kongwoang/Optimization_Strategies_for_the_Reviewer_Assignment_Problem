from __future__ import annotations
import os, time, copy, math, random
from typing import Any, List, Tuple


def read_instance(path: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        tok = [int(x) for line in f for x in line.strip().split()]
    N, M, b = tok[:3]
    idx, L = 3, []
    for _ in range(N):
        k = tok[idx]; idx += 1
        revs = tok[idx:idx + k]; idx += k
        L.append([r - 1 for r in revs])           
    return N, M, b, L


OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b if abs(b) > 1e-9 else a,
    'min': lambda a, b: a if a < b else b,
    'max': lambda a, b: a if a > b else b,
}
OP_NAMES   = list(OPS.keys())
TERM_NAMES = ['load', 'slack', 'deg', 'candCnt', 'rand', 'const']
MAX_DEPTH  = 4


def run_gp(N: int, M: int, B: int, L: List[List[int]],
           *,
           pop_size=200,
           max_generations=40,
           seed=42,
           bad_init=False,
           time_limit_s: float | None = None) -> Tuple[int, int]:

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
            _, name, lft, rgt = node
            return OPS[name](eval_tree(lft, r, i, loads),
                             eval_tree(rgt, r, i, loads))
        _, name, val = node
        if name == 'load':    return loads[rev_idx(r)]
        if name == 'slack':   return GLOBAL_QUOTA - loads[rev_idx(r)]
        if name == 'deg':     return REVIEWER_DEG[rev_idx(r)]
        if name == 'candCnt': return len(L[pap_idx(i)])
        if name == 'rand':    return random.random()
        if name == 'const':   return val
        raise ValueError


    def gen_full_tree(d):
        if d == 0:
            t = random.choice(TERM_NAMES)
            return ('term', t, rand_const() if t == 'const' else None)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_full_tree(d-1), gen_full_tree(d-1))

    def gen_grow_tree(d):
        if d == 0 or (d > 0 and random.random() < .3):
            t = random.choice(TERM_NAMES)
            return ('term', t, rand_const() if t == 'const' else None)
        op = random.choice(OP_NAMES)
        return ('op', op, gen_grow_tree(d-1), gen_grow_tree(d-1))

    def gen_bad_tree():
        return ('term', 'const', random.uniform(50, 100))

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
        viol  = 0
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
        return max(loads), viol


    if bad_init:
        pop = [gen_bad_tree() for _ in range(pop_size)]
    else:
        depths = range(2, MAX_DEPTH + 1)
        per_d  = pop_size // (2 * len(depths)) or 1
        pop = []
        for d in depths:
            pop.extend(gen_full_tree(d) for _ in range(per_d))
            pop.extend(gen_grow_tree(d) for _ in range(per_d))
        while len(pop) < pop_size:
            d = random.choice(tuple(depths))
            pop.append(gen_grow_tree(d))

    fit  = [build_assignment(t) for t in pop]
    best = min(zip(pop, fit), key=lambda x: x[1])
    TOUR, CXPB, MUTPB = 5, .9, .1


    gen = 0
    while gen < max_generations:
        if time_limit_s and (time.time() - start_time) >= time_limit_s:
            break
        gen += 1


        def select():
            idxs = random.sample(range(pop_size), TOUR)
            return copy.deepcopy(min(idxs, key=lambda j: fit[j]) and pop[min(idxs, key=lambda j: fit[j])])
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = select(), select()
            child  = crossover(p1, p2) if random.random() < CXPB else copy.deepcopy(p1)
            if random.random() < MUTPB:
                child = mutate(child)
            new_pop.append(child)
        pop  = new_pop
        fit  = [build_assignment(t) for t in pop]
        cand = min(zip(pop, fit), key=lambda x: x[1])
        if cand[1] < best[1]:
            best = (copy.deepcopy(cand[0]), cand[1])

    return best[1]   

def write_result(path, n, m, obj, runtime_ms):
    with open(path, "w") as f:
        f.write(f"{os.path.basename(path)}\n")
        f.write(f"n = {n}\n")
        f.write(f"m = {m}\n")
        f.write(f"Objective Value: {obj}\n")
        f.write(f"{runtime_ms} ms\n")


def main():
    root      = os.getcwd()
    inst_dir  = os.path.join(root, "instances")
    res_dir   = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    files = [f for f in os.listdir(inst_dir) if f.endswith(".txt")]

    for fname in files:
        in_path  = os.path.join(inst_dir, fname)
        out_name = f"[GP] {fname}"
        out_path = os.path.join(res_dir, out_name)

        print(f"Đang xử lý: {fname} → {out_name}")
        N, M, B, L = read_instance(in_path)

        start = time.time()
        max_load, _ = run_gp(N, M, B, L,
                             seed=42,
                             time_limit_s=600)       
        runtime_ms = int((time.time() - start) * 1000)

        write_result(out_path, N, M, max_load, runtime_ms)

if __name__ == "__main__":
    main()

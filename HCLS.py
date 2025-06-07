
import os, time, random
from collections import defaultdict
from typing import List, Dict, Tuple

class LocalSearch:
    def __init__(self, N: int, M: int, b: int):
        self.N = N
        self.M = M
        self.b = b

    def get_load(self, sol) -> defaultdict:
        loads = defaultdict(int)
        for r_id in range(1, self.M + 1):
            loads[r_id] = 0
        for p_assign in sol:
            for rev_id in p_assign:
                loads[rev_id] += 1
        return loads

    def solve(self, L: Dict[int, List[int]]) -> Tuple[List[List[int]], List[int]]:
        # Khởi tạo ngẫu nhiên
        cur_sol = []
        for p_idx in range(self.N):
            avail_revs = L[p_idx + 1]
            if len(avail_revs) < self.b:
                raise ValueError(f"Not enough reviewers for paper {p_idx + 1}")
            cur_sol.append(random.sample(avail_revs, self.b))

        # Local search hill-climbing
        while True:
            found = False
            cur_loads = self.get_load(cur_sol)
            if not cur_loads:
                break
            cur_max = max(cur_loads.values())
            search = max(cur_loads, key=cur_loads.get)          # reviewer nặng nhất

            # các paper mà reviewer này đang chấm
            search_papers = [i for i, assign in enumerate(cur_sol) if search in assign]

            for p_idx in search_papers:
                p_assigned = set(cur_sol[p_idx])
                p_eligible = set(L[p_idx + 1])
                new_revs   = p_eligible - p_assigned

                # thử thay thế reviewer nặng bằng reviewer nhẹ
                for replace in new_revs:
                    if cur_loads.get(replace, 0) < cur_max - 1:
                        cur_sol[p_idx].remove(search)
                        cur_sol[p_idx].append(replace)
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        final_loads = list(self.get_load(cur_sol).values())
        return cur_sol, final_loads  # list[int]

# ────────────────────────────────────────────────────────────────
#  I/O helpers
# ────────────────────────────────────────────────────────────────
def read_instance(path: str) -> Tuple[int, int, int, Dict[int, List[int]]]:
    """
    Đọc file .txt theo định dạng:
    n m b
    k reviewer1 reviewer2 ...
    (reviewer đánh số 1-based)
    """
    with open(path, "r", encoding="utf-8") as f:
        n, m, b = map(int, f.readline().split())
        L: Dict[int, List[int]] = {}
        for p in range(1, n + 1):
            parts = list(map(int, f.readline().split()))
            k, reviewers = parts[0], parts[1:]
            assert len(reviewers) == k
            L[p] = reviewers
    return n, m, b, L

def write_result(out_path: str, n: int, m: int, obj: int, runtime_ms: int):
    with open(out_path, "w") as f:
        f.write(f"{os.path.basename(out_path)}\n")
        f.write(f"n = {n}\n")
        f.write(f"m = {m}\n")
        f.write(f"Objective Value: {obj} FEASIBLE\n")
        f.write(f"{runtime_ms} ms\n")

# ────────────────────────────────────────────────────────────────
#  Batch runner
# ────────────────────────────────────────────────────────────────
def main():
    random.seed(42)                       # tái lập kết quả
    root      = os.getcwd()
    inst_dir  = os.path.join(root, "instances")
    res_dir   = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    files = [f for f in os.listdir(inst_dir) if f.endswith(".txt")]

    for fname in files:
        in_path  = os.path.join(inst_dir, fname)
        out_name = f"[LocalSearch] {fname}"
        out_path = os.path.join(res_dir, out_name)

        print(f"Đang xử lý: {fname} → {out_name}")
        n, m, b, L = read_instance(in_path)

        start = time.time()
        _, loads = LocalSearch(n, m, b).solve(L)
        runtime_ms = int((time.time() - start) * 1000)

        max_load = max(loads) if loads else 0
        write_result(out_path, n, m, max_load, runtime_ms)

if __name__ == "__main__":
    main()

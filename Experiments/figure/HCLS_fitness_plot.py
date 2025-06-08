
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1)  ĐẶT ĐƯỜNG DẪN INPUT Ở ĐÂY  ⬇⬇⬇
FILE_PATH = "instances\\Adversarial_2000_150_5.txt"        # <— thay đổi nếu cần

# -------------------------------------------------------------


# ----------------- Local-Search core -----------------
class LocalSearch:
    def __init__(self, N: int, M: int, b: int):
        self.N, self.M, self.b = N, M, b

    def _get_load(self, sol):
        load = defaultdict(int)
        for r in range(1, self.M + 1):
            load[r] = 0
        for assign in sol:
            for r in assign:
                load[r] += 1
        return load

    def solve(self, L):
        sol = [random.sample(L[p], self.b) for p in range(1, self.N + 1)]
        hist = []

        while True:
            load = self._get_load(sol)
            max_load = max(load.values())
            hist.append(max_load)

            worst = max(load, key=load.get)
            papers = [i for i, a in enumerate(sol) if worst in a]

            improved = False
            for p_idx in papers:
                cur   = set(sol[p_idx])
                avail = set(L[p_idx + 1]) - cur
                for cand in avail:
                    if load[cand] < max_load - 1:
                        sol[p_idx].remove(worst)
                        sol[p_idx].append(cand)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return sol, hist


# ----------------- I/O utils -----------------
def read_instance(path):
    with open(path, encoding="utf-8") as f:
        toks = list(map(int, f.read().strip().split()))
    it = iter(toks)
    N, M, b = next(it), next(it), next(it)
    L = {}
    for p in range(1, N + 1):
        k = next(it)
        L[p] = [next(it) for _ in range(k)]
    return N, M, b, L


def plot_hist(hist):
    plt.figure()
    plt.plot(range(len(hist)), hist, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Max reviewer load")
    plt.title("Fitness (max-load) over iterations")
    plt.grid(True)
    plt.savefig('Experiments\\figure\\HCLS_fitness_plot.png', dpi=300, bbox_inches='tight')

# ----------------- main -----------------
if __name__ == "__main__":
    N, M, b, L = read_instance(FILE_PATH)
    ls = LocalSearch(N, M, b)
    sol, hist = ls.solve(L)

    print(f"Final max-load: {hist[-1]}")
    for p_id, revs in enumerate(sol, 1):
        print(f"Paper {p_id}: {revs}")

    plot_hist(hist)
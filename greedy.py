import os
import time

# ------------------------------------------------------------------
# Data-model classes
# ------------------------------------------------------------------
class Reviewer:
    def __init__(self, ID: int):
        self.ID = ID         # 1-based ID
        self.papers = []     # các Paper mà reviewer này có thể chấm

    def __str__(self) -> str:
        ans = f"Reviewer {self.ID}\n"
        for paper in self.papers:
            ans += f"\tPaper {paper.ID}\n"
        return ans


class Paper:
    def __init__(self, ID: int):
        self.ID = ID               # 1-based ID
        self.reviewers = []        # list Reviewer sẵn sàng chấm
        self.sol = []              # list Reviewer thực sự được gán

    def __str__(self) -> str:
        ans = f"Paper {self.ID}\n"
        for reviewer in self.reviewers:
            ans += f"\tReviewer {reviewer.ID}\n"
        return ans


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------
def InputFile(filename: str):
    """Đọc dữ liệu và trả về n, m, b, list[list[int]] (reviewer IDs 0-based)."""
    with open(filename, "r") as f:
        n, m, b = map(int, f.readline().split())
        prefs = []
        for _ in range(n):
            parts = list(map(int, f.readline().split()))
            reviewers = [r - 1 for r in parts[1:]]  # chuyển 0-based
            prefs.append(reviewers)
        return n, m, b, prefs


def build_objects(n: int, m: int, paper_prefs):
    """Tạo danh sách Paper / Reviewer từ raw prefs."""
    reviewers = [Reviewer(i + 1) for i in range(m)]   # 1-based ID
    papers = [Paper(i + 1) for i in range(n)]

    for p_idx, pref in enumerate(paper_prefs):
        for r_idx in pref:
            r = reviewers[r_idx]
            papers[p_idx].reviewers.append(r)
            r.papers.append(papers[p_idx])
    return papers, reviewers


# ------------------------------------------------------------------
# Greedy solver
# ------------------------------------------------------------------
class Solver:
    def __init__(self, papers, reviewers, b: int):
        self.papers = papers
        self.reviewers = reviewers
        self.b = b

    def solve(self):
        # Sắp xếp paper theo số reviewer khả dụng (dễ trước, khó sau)
        self.papers.sort(key=lambda p: len(p.reviewers))

        # đếm số bài mỗi reviewer đã nhận
        load = {rev.ID: 0 for rev in self.reviewers}

        # Gán lặp b vòng để bảo đảm mỗi paper nhận đủ b reviewer
        for _ in range(self.b):
            for paper in self.papers:
                # Chọn reviewer “nhẹ” nhất & ít lựa chọn nhất
                reviewer = min(
                    paper.reviewers,
                    key=lambda rev: (load[rev.ID], len(rev.papers))
                )
                paper.sol.append(reviewer)
                load[reviewer.ID] += 1
                paper.reviewers.remove(reviewer)

        # Trả về tải tối đa để in báo cáo
        return max(load.values()) if load else 0


# ------------------------------------------------------------------
# Chạy một instance + ghi file kết quả
# ------------------------------------------------------------------
def run_instance(input_path: str, output_path: str):
    n, m, b, prefs = InputFile(input_path)
    papers, reviewers = build_objects(n, m, prefs)

    solver = Solver(papers, reviewers, b)
    objective = solver.solve()  # max load

    with open(output_path, "w") as f:
        f.write(f"{os.path.basename(output_path)}\n")
        f.write(f"n = {n}\n")
        f.write(f"m = {m}\n")
        f.write(f"Objective Value: {objective}\n") 


# ------------------------------------------------------------------
# Batch runner: giống cấu trúc trước
# ------------------------------------------------------------------
def main():
    cur_dir = os.getcwd()
    instances_dir = os.path.join(cur_dir, "instances")
    results_dir   = os.path.join(cur_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    input_files = [f for f in os.listdir(instances_dir) if f.endswith(".txt")]

    for fname in input_files:
        in_path  = os.path.join(instances_dir, fname)
        out_name = f"[Greedy] {fname}"
        out_path = os.path.join(results_dir, out_name)
        print(f"Đang xử lý: {fname} → {out_name}")

        start = time.time()
        run_instance(in_path, out_path)
        runtime_ms = int((time.time() - start) * 1000)

        # Ghi thêm thời gian chạy
        with open(out_path, "a") as f:
            f.write(f"{runtime_ms} ms\n")


if __name__ == "__main__":
    main()

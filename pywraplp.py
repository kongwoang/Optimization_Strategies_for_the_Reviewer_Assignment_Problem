from ortools.linear_solver import pywraplp
import os
import time

def InputFile(filename):
    """Đọc dữ liệu từ file và điều chỉnh chỉ số reviewer về dạng 0-based."""
    with open(filename, 'r') as f:
        n, m, b = map(int, f.readline().split())
        paper_preferences = []
        for _ in range(n):
            parts = list(map(int, f.readline().split()))
            reviewers = [r - 1 for r in parts[1:]]  # Chuyển về chỉ số 0-based
            paper_preferences.append(reviewers)
        return n, m, b, paper_preferences

def solve_reviewers_assignment_ilp(n, m, b, paper_prefs, output_file):
    """Giải bài toán phân công reviewer và ghi kết quả vào file output."""
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return
    solver.set_time_limit(600000)
    # Khai báo biến chỉ cho các cặp (paper, reviewer) hợp lệ
    x = {}
    reviewer_papers = [[] for _ in range(m)]
    for i in range(n):
        for r in paper_prefs[i]:
            x[i, r] = solver.IntVar(0, 1, f'x_{i}_{r}')
            reviewer_papers[r].append(i)

    # Biến tải tối đa
    max_load = solver.IntVar(0, n, 'max_load')

    # Ràng buộc: mỗi paper có đúng b reviewer
    for i in range(n):
        solver.Add(sum(x[i, r] for r in paper_prefs[i]) == b)

    # Ràng buộc: max_load >= tải của mỗi reviewer
    for r in range(m):
        if reviewer_papers[r]:
            solver.Add(max_load >= sum(x[i, r] for i in reviewer_papers[r]))

    # Mục tiêu: tối thiểu hóa tải tối đa
    solver.Minimize(max_load)

    # Giải bài toán
    status = solver.Solve()

    # Ghi kết quả vào file output
    with open(output_file, 'w') as f:
        if status == pywraplp.Solver.OPTIMAL:
            f.write(f"{str(output_file).split("\\")[-1]}\n")
            f.write(f"n = {n}\nm = {m}\n")
            f.write(f"Objective Value: {int(max_load.solution_value())} OPTIMAL\n")
        elif status == pywraplp.Solver.FEASIBLE:
            f.write(f"{str(output_file).split("\\")[-1]}\n")
            f.write(f"n = {n}\nm = {m}\n")
            f.write(f"Objective Value: {int(max_load.solution_value())} FEASIBLE\n")     
        else:
            f.write("No solution found.\n")

def main():
    """Hàm chính để chạy solver trên tất cả file .txt trong thư mục 'instances' và ghi kết quả vào 'results'."""
    # Lấy đường dẫn thư mục hiện tại
    current_dir = os.getcwd()
    instances_dir = os.path.join(current_dir, 'instances')
    results_dir = os.path.join(current_dir, 'results')

    # Tạo thư mục 'results' nếu chưa có
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Kiểm tra thư mục 'instances' có tồn tại không
    if not os.path.exists(instances_dir):
        print(f"Thư mục '{instances_dir}' không tồn tại.")
        return

    # Lấy danh sách các file .txt trong thư mục 'instances'
    input_files = [f for f in os.listdir(instances_dir) if f.endswith('.txt')]

    # Xử lý từng file
    for filename in input_files:
        input_path = os.path.join(instances_dir, filename)
        output_filename = f"[ILP_Ortools] {filename}"
        output_path = os.path.join(results_dir, output_filename)
        print(f"Đang xử lý file: {filename} -> {output_filename}")
        try:
            n, m, b, paper_preferences = InputFile(input_path)
            start_time = time.time()  # Ghi lại thời điểm bắt đầu
            solve_reviewers_assignment_ilp(n, m, b, paper_preferences, output_path)
            end_time = time.time()    # Ghi lại thời điểm kết thúc
            run_time = end_time - start_time  # Tính thời gian chạy
            # Ghi thêm thời gian chạy vào file output
            with open(output_path, 'a') as f:
                f.write(f"{int(run_time * 1000)} ms\n")
        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")

if __name__ == "__main__":
    main()
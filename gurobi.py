import gurobipy as gp
from gurobipy import GRB
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
    try:
        # Tạo model Gurobi
        model = gp.Model("reviewers_assignment")

        # Tạo list các cặp (i, r) hợp lệ
        valid_pairs = [(i, r) for i in range(n) for r in paper_prefs[i]]

        # Khai báo biến x cho các cặp hợp lệ
        x = model.addVars(valid_pairs, vtype=GRB.BINARY, name="x")

        # Khai báo biến max_load
        max_load = model.addVar(vtype=GRB.INTEGER, name="max_load")

        # Ràng buộc: mỗi paper có đúng b reviewer
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, r] for r in paper_prefs[i]) == b)

        # Tạo list reviewer_papers
        reviewer_papers = [[] for _ in range(m)]
        for i in range(n):
            for r in paper_prefs[i]:
                reviewer_papers[r].append(i)

        # Ràng buộc: max_load >= tải của mỗi reviewer
        for r in range(m):
            if reviewer_papers[r]:
                model.addConstr(max_load >= gp.quicksum(x[i, r] for i in reviewer_papers[r]))

        # Đặt mục tiêu: tối thiểu hóa max_load
        model.setObjective(max_load, GRB.MINIMIZE)

        # Giải model
        model.optimize()

        # Ghi kết quả vào file output
        with open(output_file, 'w') as f:
            if model.status == GRB.OPTIMAL:
                f.write(f"{str(output_file).split("\\")[-1]}\n")
                f.write(f"n = {n}\nm = {m}\n")
                f.write(f"Objective Value: {int(max_load.x)}\n")
            else:
                f.write("No solution found.\n")
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Error: {e}")

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
        output_filename = f"[ILP_gurobi] {filename}"
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
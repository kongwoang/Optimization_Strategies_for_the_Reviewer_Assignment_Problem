import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Dữ liệu đầy đủ từ nội dung bạn cung cấp
data_str = """
sample,n,m,b,ILP_Ortools_objective,ILP_Ortools_time_ms,CP_Ortools_objective,CP_Ortools_time_ms,ILP_gurobi_objective,ILP_gurobi_time_ms
Adversarial 0,50,20,2,5,11,5,16,5,5
Exponential 0,50,20,2,5,16,5,22,5,9
Gaussian 0,50,20,2,5,15,5,20,5,6
Poisson 0,50,20,2,5,15,5,26,5,6
Uniform 0,50,20,2,5,14,5,18,5,7
Adversarial 1,100,50,3,7,41,7,61,7,18
Exponential 1,100,50,3,6,57,6,87,6,14
Gaussian 1,100,50,3,6,58,6,98,6,13
Poisson 1,100,50,3,6,60,6,61,6,14
Uniform 1,100,50,3,6,61,6,83,6,16
Adversarial 2,200,100,3,7,96,7,110,7,39
Exponential 2,200,100,3,6,137,6,162,6,47
Gaussian 2,200,100,3,6,111,6,206,6,32
Poisson 2,200,100,3,6,109,6,161,6,37
Uniform 2,200,100,3,6,174,6,245,6,37
hustack3,300,30,3,30,74,30,175,30,45
Adversarial 3,500,350,4,6,498,6,579,6,353
Exponential 3,500,350,4,6,1016,6,1603,6,509
Gaussian 3,500,350,4,6,691,6,1004,6,827
Poisson 3,500,350,4,6,585,6,1101,6,451
Uniform 3,500,350,4,6,1385,6,2786,6,766
Adversarial 4,800,500,5,9,1557,9,4009,9,1129
Exponential 4,800,500,5,8,4826,8,7664,8,335
Gaussian 4,800,500,5,9,1716,9,599903,9,1029
Poisson 4,800,500,5,8,1404,8,16425,8,319
Uniform 4,800,500,5,8,2847,8,16246,8,361
Adversarial 5,1000,700,5,8,3203,8,74913,8,1159
Exponential 5,1000,700,5,8,4715,8,600203,8,3272
Gaussian 5,1000,700,5,8,3946,8,498038,8,2895
Poisson 5,1000,700,5,8,2928,8,599955,8,2825
Uniform 5,1000,700,5,8,48042,8,600207,8,2940
Adversarial 6,2000,900,5,12,271164,12,600255,12,6162
Exponential 6,2000,900,5,26,617995,12,600327,12,9439
Gaussian 6,2000,900,5,12,342269,12,600287,12,5133
Poisson 6,2000,900,5,12,330627,12,600304,12,5253
Uniform 6,2000,900,5,24,760219,12,600390,12,10229
Adversarial 7,5000,2000,6,33,833872,16,600796,16,54849
Exponential 7,5000,2000,6,30,898355,15,601621,15,14820
Gaussian 7,5000,2000,6,29,848155,16,600765,16,21996
Poisson 7,5000,2000,6,30,879356,16,600721,16,32804
Uniform 7,5000,2000,6,28,934078,15,601115,15,9192
Adversarial 8,10000,4000,6,35,872372,16,601729,16,182996
Exponential 8,10000,4000,6,28,885106,15,602163,15,61932
Gaussian 8,10000,4000,6,29,855337,16,601646,16,177038
Poisson 8,10000,4000,6,30,884789,16,601885,16,177062
Uniform 8,10000,4000,6,30,884006,15,602582,15,59760
Adversarial 9,20000,9000,6,34,836491,15,565285,15,477977
Exponential 9,20000,9000,6,30,921194,14,604697,14,2409530
Gaussian 9,20000,9000,6,28,800141,14,568291,14,774545
Poisson 9,20000,9000,6,31,844061,14,578841,14,562033
Uniform 9,20000,9000,6,30,997628,14,604556,14,2265356
"""

# Đọc dữ liệu từ chuỗi (mô phỏng tệp CSV)
df = pd.read_csv(StringIO(data_str))

# Nhóm theo (n, m, b) và tính trung bình thời gian
grouped_df = df.groupby(['n', 'm', 'b']).agg({
    'ILP_Ortools_time_ms': 'mean',
    'CP_Ortools_time_ms': 'mean',
    'ILP_gurobi_time_ms': 'mean'
}).reset_index()


# Sắp xếp theo n để trục hoành có thứ tự
grouped_df = grouped_df.sort_values('n')

# Tạo nhãn trục hoành từ các bộ (n, m, b)
labels = [f"{int(row['n'])}" for _, row in grouped_df.iterrows()]

# Trích xuất thời gian trung bình
ILP_Ortools_times = grouped_df['ILP_Ortools_time_ms']
CP_Ortools_times = grouped_df['CP_Ortools_time_ms']
ILP_gurobi_times = grouped_df['ILP_gurobi_time_ms']

# Tạo chỉ số cho trục hoành
x = np.arange(len(labels))

# Vẽ biểu đồ đường
plt.figure(figsize=(15,10), dpi=300)
plt.plot(x, ILP_Ortools_times, marker='o', label='OR-Tools pywraplp', alpha=0.7)
plt.plot(x, CP_Ortools_times, marker='s', label='OR-Tools CP-SAT', alpha=0.7)
plt.plot(x, ILP_gurobi_times, marker='^', label='Gurobi', alpha=0.7)

# Đặt thang đo logarit cho trục tung
plt.yscale('log')

# Đặt nhãn trục hoành, nghiêng 45 độ, font lớn
plt.xticks(x, labels, rotation=0, ha='right', fontsize=20)

# Đặt nhãn và tiêu đề với font lớn
plt.xlabel('#Papers', fontsize=20)
plt.ylabel('Time (ms)', fontsize=20)
# plt.title('Integer Linear Programming Average Execute Time', fontsize=16)
plt.legend(fontsize=20)
plt.grid(True)

# Điều chỉnh bố cục để nhãn không bị cắt
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('figure\\ILP_plot.png', dpi=300, bbox_inches='tight')
# plt.show()
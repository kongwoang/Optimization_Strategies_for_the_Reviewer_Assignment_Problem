import pandas as pd
import matplotlib.pyplot as plt

# 1) Đọc dữ liệu
df = pd.read_csv("Experiments\\figure\\GP_results.csv")      # chỉnh đường dẫn nếu cần

# 2) Tính thời gian trung bình theo n
avg_times = (df.groupby("n")["GP_time_ms"]
               .mean()
               .reset_index()
               .sort_values("n"))

# 3) Vẽ biểu đồ
plt.figure()
plt.plot(avg_times["n"],
         avg_times["GP_time_ms"],
         marker="o")
plt.xlabel("Problem size n (number of papers)")
plt.ylabel("Average running time (ms)")

plt.grid(True)
plt.tight_layout()


# Lưu biểu đồ
plt.savefig('Experiments\\figure\\GP_time_plot.png', dpi=300, bbox_inches='tight')

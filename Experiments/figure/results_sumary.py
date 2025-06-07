import csv
import re
from pathlib import Path

# -------- cấu hình --------
RESULTS_DIR = Path("results")
OUT_CSV     = Path("summary.csv")

FILE_RE  = re.compile(r"\[(.*?)\]\s*(.*?)\.txt$", re.I)        # [+method+] sample.txt
OBJ_RE   = re.compile(r"Objective Value:\s*([+-]?\d+(?:\.\d+)?)\s+(\w+)", re.I)
TIME_RE  = re.compile(r"(\d+)\s*ms", re.I)
N_RE     = re.compile(r"n\s*=\s*(\d+)", re.I)
M_RE     = re.compile(r"m\s*=\s*(\d+)", re.I)
NAME_NM  = re.compile(r"_(\d+?)_(\d+?)_")                      # fallback lấy n,m từ tên

rows, methods = {}, set()

for path in RESULTS_DIR.glob("*.txt"):
    m = FILE_RE.match(path.name)
    if not m:
        continue
    method, sample = m.groups()
    methods.add(method)

    with path.open(encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    # n, m
    n = int(N_RE.search(txt).group(1)) if N_RE.search(txt) else None
    mval = int(M_RE.search(txt).group(1)) if M_RE.search(txt) else None
    if n is None or mval is None:
        nm = NAME_NM.search(sample)
        if nm:
            n, mval = map(int, nm.groups())

    # objective & status
    obj_match = OBJ_RE.search(txt)
    obj_val, is_opt = (None, None)
    if obj_match:
        obj_val = obj_match.group(1)
        is_opt  = obj_match.group(2).upper() == "OPTIMAL"

    # time
    t_match = TIME_RE.search(txt)
    t_ms = int(t_match.group(1)) if t_match else None

    # khởi tạo hàng nếu cần
    row = rows.setdefault(sample, {"sample": sample, "n": n, "m": mval})
    # đảm bảo ba trường cho mọi method
    for key in ("objective", "time_ms", "optimal"):
        row.setdefault(f"{method}_{key}", None)

    row[f"{method}_objective"] = obj_val
    row[f"{method}_time_ms"]   = t_ms
    row[f"{method}_optimal"]   = is_opt

# -------- ghi CSV --------
fieldnames = ["sample", "n", "m"]
for method in sorted(methods):
    fieldnames += [f"{method}_objective", f"{method}_time_ms", f"{method}_optimal"]

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for sample in sorted(rows):
        writer.writerow(rows[sample])

print(f"✅  Đã ghi {len(rows)} mẫu, {len(methods)} phương pháp → {OUT_CSV}")

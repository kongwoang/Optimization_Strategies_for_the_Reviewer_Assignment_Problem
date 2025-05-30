import csv
import os
import re
from pathlib import Path

# ----- cấu hình -----
RESULTS_DIR = Path("results")
OUTPUT_FILE = Path("summary.csv")

# regex bắt tên file và thông tin bên trong
FILENAME_RE = re.compile(r"\[(.*?)\]\s*(.*?)\.txt$", re.I)   # [CP_Ortools] Adversarial_50_20_2.txt
NM_RE       = re.compile(r"_(\d+?)_(\d+?)_")                 # _50_20_
OBJ_RE      = re.compile(r"Objective Value:\s*([+-]?\d+(?:\.\d+)?)\s+(\w+)", re.I)
TIME_RE     = re.compile(r"(\d+)\s*ms", re.I)

# ánh xạ tên phương pháp -> tiền tố cột trong CSV
METHOD_MAP = {
    "CP_Ortools":  "CP",
    "ILP_Ortools": "ILP",
}

# ----- gom dữ liệu -----
rows = {}          # key = sample name, value = dict(các trường)
for path in RESULTS_DIR.glob("*.txt"):
    mo = FILENAME_RE.search(path.name)
    if not mo:
        continue                                        # bỏ qua tên lạ
    method_raw, sample = mo.groups()                   # CP_Ortools  |  Adversarial_50_20_2
    method = METHOD_MAP.get(method_raw)
    if method is None:                                 # nếu là phương pháp khác, đơn giản bỏ qua
        continue

    # đọc file
    with path.open(encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # lấy n, m (ưu tiên dòng “n = …”), nếu không có thì bóc từ tên sample
    n, m = None, None
    n_line = re.search(r"n\s*=\s*(\d+)", content, re.I)
    m_line = re.search(r"m\s*=\s*(\d+)", content, re.I)
    if n_line and m_line:
        n, m = int(n_line.group(1)), int(m_line.group(1))
    else:                                              # fallback: lấy trong tên sample
        mo_nm = NM_RE.search(sample)
        if mo_nm:
            n, m = map(int, mo_nm.groups())

    # lấy objective & status (OPTIMAL/HIGHER/…)
    mo_obj = OBJ_RE.search(content)
    obj_val, status = (None, None)
    if mo_obj:
        obj_val  = mo_obj.group(1)
        status   = mo_obj.group(2).upper() == "OPTIMAL"   # True / False

    # lấy thời gian
    mo_time = TIME_RE.search(content)
    time_ms = int(mo_time.group(1)) if mo_time else None

    # cập nhật dictionary
    row = rows.setdefault(sample, {
        "sample": sample,
        "n": n, "m": m,
        "CP_objective": None,   "CP_time_ms": None,   "CP_optimal": None,
        "ILP_objective": None,  "ILP_time_ms": None,  "ILP_optimal": None,
    })
    row[f"{method}_objective"] = obj_val
    row[f"{method}_time_ms"]   = time_ms
    row[f"{method}_optimal"]   = status

# ----- ghi CSV -----
fieldnames = [
    "sample", "n", "m",
    "CP_objective",  "CP_time_ms",  "CP_optimal",
    "ILP_objective", "ILP_time_ms", "ILP_optimal",
]
with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for sample in sorted(rows):          # giữ thứ tự rõ ràng
        writer.writerow(rows[sample])

print(f"✅  Đã ghi {len(rows)} dòng vào {OUTPUT_FILE}")

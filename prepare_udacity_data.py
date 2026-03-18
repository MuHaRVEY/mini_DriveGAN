import os
import csv
import shutil
import pandas as pd

# 1. 원본 Udacity 데이터 위치
# 예시:
# raw_data/
#   ├─ IMG/
#   └─ driving_log.csv
RAW_ROOT = "raw_data"
RAW_IMG_DIR = os.path.join(RAW_ROOT, "IMG")
RAW_CSV_PATH = os.path.join(RAW_ROOT, "driving_log.csv")

# 2. 프로젝트용 출력 위치
OUT_FRAME_DIR = os.path.join("data", "frames")
OUT_CSV_PATH = os.path.join("data", "labels.csv")

os.makedirs(OUT_FRAME_DIR, exist_ok=True)

# Udacity behavioral cloning 계열 csv 읽기
df = pd.read_csv(RAW_CSV_PATH)

# 컬럼 이름 확인용 출력
print("Columns:", df.columns.tolist())

# 보통 center / steering 컬럼이 존재
# 일부 데이터셋은 center 경로에 공백이 들어가 있을 수 있음
# 경로가 전체 경로일 수도 있고 파일명만 있을 수도 있음
rows = []
new_index = 1

for _, row in df.iterrows():
    center_path = str(row["center"]).strip()
    steering = float(row["steering"])

    # 파일명만 추출
    filename_only = os.path.basename(center_path)

    # 원본 이미지 경로
    src_path = os.path.join(RAW_IMG_DIR, filename_only)

    # 만약 center 값이 이미 IMG/... 형태면 basename으로 충분
    if not os.path.exists(src_path):
        # 혹시 center_path 자체가 유효 경로일 경우 대비
        alt_path = center_path
        if os.path.exists(alt_path):
            src_path = alt_path
        else:
            print(f"Skip: image not found -> {center_path}")
            continue

    new_filename = f"{new_index:06d}.jpg"
    dst_path = os.path.join(OUT_FRAME_DIR, new_filename)

    shutil.copy2(src_path, dst_path)
    rows.append([new_filename, steering])

    new_index += 1

with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "steering"])
    writer.writerows(rows)

print(f"Saved {len(rows)} frames to {OUT_FRAME_DIR}")
print(f"Saved labels to {OUT_CSV_PATH}")
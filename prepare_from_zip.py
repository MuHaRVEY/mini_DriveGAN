import os
import csv
import zipfile
from io import TextIOWrapper
from PIL import Image
import pandas as pd

ZIP_PATH = "data_from_kaggle.zip"   # 네 zip 파일 이름
DATASET_NAME = "self_driving_car_dataset_jungle"   # jungle 또는 make
STRIDE = 5   # 5면 5장마다 1장만 사용

OUT_FRAME_DIR = os.path.join("data", "frames")
OUT_CSV_PATH = os.path.join("data", "labels.csv")

os.makedirs(OUT_FRAME_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    csv_path = f"{DATASET_NAME}/driving_log.csv"

    # 헤더 없는 CSV라서 header=None 사용
    with z.open(csv_path) as f:
        df = pd.read_csv(f, header=None)

    # 컬럼 이름 강제로 지정
    df.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

    rows = []
    new_index = 1

    for i in range(0, len(df), STRIDE):
        row = df.iloc[i]

        center_path = str(row["center"]).strip()
        steering = float(row["steering"])

        # CSV 안에는 윈도우 절대경로가 들어있으므로 파일명만 추출
        filename_only = os.path.basename(center_path)

        # zip 내부 실제 이미지 경로
        zip_img_path = f"{DATASET_NAME}/IMG/{filename_only}"

        if zip_img_path not in z.namelist():
            print(f"Skip: {zip_img_path}")
            continue

        new_filename = f"{new_index:06d}.jpg"
        out_img_path = os.path.join(OUT_FRAME_DIR, new_filename)

        # zip에서 이미지 추출해서 저장
        with z.open(zip_img_path) as img_file:
            img = Image.open(img_file).convert("RGB")
            img.save(out_img_path)

        rows.append([new_filename, steering])
        new_index += 1

with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "steering"])
    writer.writerows(rows)

print(f"Done. Saved {len(rows)} images.")
print(f"Frames -> {OUT_FRAME_DIR}")
print(f"Labels -> {OUT_CSV_PATH}")
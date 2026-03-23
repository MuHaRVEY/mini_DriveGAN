import os
import pandas as pd #csv 파일
from PIL import Image #이미지를 여는데 필요하다고 함.

import torch
from torch.utils.data import Dataset# pytorch의 데이터셋
from torchvision import transforms #이미지 크기, tensor 변환


class DrivingFrameDataset(Dataset): #데이터셋을 상속받아 사용
    def __init__(self, csv_path, frame_dir, image_size=128, num_input_frames = 4):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. Expected columns: frame, steering"
            )

        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(
                f"Frame directory not found: {frame_dir}"
            )

        if not any(os.scandir(frame_dir)):
            raise ValueError(
                f"Frame directory is empty: {frame_dir}. Add image files before training."
            )

        try:
            self.data = pd.read_csv(csv_path) #CSV를 self.data에 저장
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"CSV is empty: {csv_path}. Add header 'frame,steering' and at least two rows."
            ) from exc

        required_columns = {"frame", "steering"}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_columns)}"
            )

        if len(self.data) < 2:
            raise ValueError(
                f"Not enough rows in CSV: {csv_path}. Need at least 2 rows for (t, t+1) pairs."
            )

        missing_frames = []
        for frame_name in self.data["frame"].astype(str):
            frame_path = os.path.join(frame_dir, frame_name)
            if not os.path.exists(frame_path):
                missing_frames.append(frame_name)
            if len(missing_frames) >= 5:
                break

        if missing_frames:
            raise FileNotFoundError(
                "Frame files listed in CSV were not found in frame_dir. "
                f"Examples: {missing_frames}"
            )

        self.frame_dir = frame_dir #프레임의 폴더 경로 slef.frame_dir저장
        self.num_input_frames = num_input_frames

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),        #Resize로 이미지를 image_size x image_size로 맞춤
            transforms.ToTensor(),                              # 0 ~ 1 범위 Tensor로 변환
        ])

    def __len__(self):
        # 최근 4프레임을 써야 하므로 시작 index를 확보해야 함
        # 예: idx=3일 때 [0,1,2,3] -> target=4
        return len(self.data) - self.num_input_frames 

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}. Valid range is 0 to {len(self) - 1}.")

        input_frames = []
        for i in range(idx, idx + self.num_input_frames):
            row = self.data.iloc[i]
            img_path = os.path.join(self.frame_dir, row["frame"])

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            input_frames.append(img)

        # [3, H, W] x 4 -> [12, H, W]
        x_t = torch.cat(input_frames, dim=0)

        # action은 마지막 입력 프레임 시점의 steering 사용
        action_row = self.data.iloc[idx + self.num_input_frames - 1]
        steering = torch.tensor([float(action_row["steering"])], dtype=torch.float32)

        # target은 그 다음 프레임
        next_row = self.data.iloc[idx + self.num_input_frames]
        next_path = os.path.join(self.frame_dir, next_row["frame"])

        x_next = Image.open(next_path).convert("RGB")
        x_next = self.transform(x_next)      
        
        """ 일단 기존 코드 킵"""
        # current_row = self.data.iloc[idx]
        # next_row = self.data.iloc[idx + 1]

        # current_path = os.path.join(self.frame_dir, current_row["frame"])
        # next_path = os.path.join(self.frame_dir, next_row["frame"])

        # x_t = Image.open(current_path).convert("RGB")
        # x_next = Image.open(next_path).convert("RGB")

        # x_t = self.transform(x_t) # 시점 t 프레임
        # x_next = self.transform(x_next) # 시점 t+1 프레임

        # steering = torch.tensor([float(current_row["steering"])], dtype=torch.float32)

        return x_t, steering, x_next
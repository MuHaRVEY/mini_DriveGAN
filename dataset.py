import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DrivingSequenceDataset(Dataset):
    def __init__(self, csv_path, frame_dir, image_size=128, seq_len=4):
        self.data = pd.read_csv(csv_path)
        self.frame_dir = frame_dir
        self.seq_len = seq_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # 입력 4장 + 정답 1장 필요
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        frames = []

        # 입력 시퀀스: x_{t-3}, x_{t-2}, x_{t-1}, x_t
        for i in range(idx, idx + self.seq_len):
            row = self.data.iloc[i]
            img_path = os.path.join(self.frame_dir, row["frame"])

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        # [T, 3, H, W]
        x_seq = torch.stack(frames, dim=0)

        # 마지막 입력 프레임 시점의 steering 사용
        action_row = self.data.iloc[idx + self.seq_len - 1]
        action = torch.tensor([float(action_row["steering"]),
                               float(action_row["throttle"]),
                               float(action_row["brake"]),
                               float(action_row["speed"])/30.0 # 속도 정규화 시도)
                               ], dtype=torch.float32)

        # 정답: 다음 프레임 x_{t+1}
        next_row = self.data.iloc[idx + self.seq_len]
        next_path = os.path.join(self.frame_dir, next_row["frame"])

        x_next = Image.open(next_path).convert("RGB")
        x_next = self.transform(x_next)

        return x_seq, action, x_next
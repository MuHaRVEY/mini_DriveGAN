import os
import torch
from torchvision.utils import save_image


def save_prediction_samples(x_seq, x_next, x_pred, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)

    # 입력 시퀀스 중 마지막 프레임만 시각화
    x_t_vis = x_seq[:, -1, :, :, :]  # [B, 3, H, W]

    comparison = torch.cat([x_t_vis[:4], x_next[:4], x_pred[:4]], dim=0)
    save_image(
        comparison,
        os.path.join(save_dir, f"epoch_{epoch:03d}.png"),
        nrow=4
    )
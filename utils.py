import os
import torch
from torchvision.utils import save_image


def save_prediction_samples(x_t, x_next, x_pred, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    
    x_t_vis = x_t[:, -3:, :, :]
    
    comparison = torch.cat([x_t_vis[:4], x_next[:4], x_pred[:4]], dim=0)
    save_image(comparison, os.path.join(save_dir, f"epoch_{epoch:03d}.png"), nrow=4)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DrivingSequenceDataset
from model import MiniDriveGANSequence
from utils import save_prediction_samples


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/labels.csv"
    frame_dir = "data/frames"

    batch_size = 8
    lr = 1e-3
    epochs = 20
    image_size = 128
    hidden_dim = 256
    seq_len = 4

    dataset = DrivingSequenceDataset(
        csv_path=csv_path,
        frame_dir=frame_dir,
        image_size=image_size,
        seq_len=seq_len
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MiniDriveGANSequence(
        hidden_dim=hidden_dim,
        action_dim=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.L1Loss()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_seq, steering, x_next in loader:
            x_seq = x_seq.to(device)         # [B, T, 3, H, W]
            steering = steering.to(device)   # [B, 1]
            x_next = x_next.to(device)       # [B, 3, H, W]

            x_pred, z_seq, z_next_pred, _, _ = model(x_seq, steering)

            loss_recon = recon_loss_fn(x_pred, x_next)

            with torch.no_grad():
                # 실제 다음 latent target:
                # 입력 시퀀스에서 첫 프레임 제거하고 x_next 추가
                next_seq = torch.cat([x_seq[:, 1:], x_next.unsqueeze(1)], dim=1)
                z_next_true_seq = model.encode_sequence(next_seq)
                z_next_true = z_next_true_seq[:, -1]

            loss_latent = nn.functional.mse_loss(z_next_pred, z_next_true)
            loss = loss_recon + 0.1 * loss_latent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            x_seq, steering, x_next = sample_batch

            x_seq = x_seq.to(device)
            steering = steering.to(device)
            x_next = x_next.to(device)

            x_pred, _, _, _, _ = model(x_seq, steering)
            save_prediction_samples(
                x_seq.cpu(),
                x_next.cpu(),
                x_pred.cpu(),
                "outputs",
                epoch
            )

        torch.save(model.state_dict(), f"checkpoints/mini_drivegan_sequence_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()
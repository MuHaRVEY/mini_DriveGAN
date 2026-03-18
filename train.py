import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DrivingFrameDataset
from model import MiniDriveGAN
from utils import save_prediction_samples


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/labels.csv"
    frame_dir = "data/frames"

    batch_size = 16
    lr = 1e-3
    epochs = 20
    image_size = 128
    latent_dim = 128

    dataset = DrivingFrameDataset(
        csv_path=csv_path,
        frame_dir=frame_dir,
        image_size=image_size
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MiniDriveGAN(latent_dim=latent_dim, action_dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.L1Loss()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_t, steering, x_next in loader:
            x_t = x_t.to(device)
            steering = steering.to(device)
            x_next = x_next.to(device)

            x_pred, z_t, z_next_pred = model(x_t, steering)

            loss_recon = recon_loss_fn(x_pred, x_next)

            # optional latent consistency target
            with torch.no_grad():
                z_next_true = model.encoder(x_next)

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
            x_t, steering, x_next = sample_batch
            x_t = x_t.to(device)
            steering = steering.to(device)
            x_next = x_next.to(device)

            x_pred, _, _ = model(x_t, steering)
            save_prediction_samples(x_t.cpu(), x_next.cpu(), x_pred.cpu(), "outputs", epoch)

        torch.save(model.state_dict(), f"checkpoints/mini_drivegan_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()
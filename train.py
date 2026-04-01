import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DrivingSequenceDataset
from model import MiniDriveGANPaperLike, Discriminator
from utils import save_prediction_samples


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/labels.csv"
    frame_dir = "data/frames"

    batch_size = 8
    lr_G = 1e-4
    lr_D = 1e-4 
    epochs = 20
    image_size = 128
    hidden_dim = 256
    theme_dim = 64
    seq_len = 4
    action_dim = 4

    dataset = DrivingSequenceDataset(
        csv_path=csv_path,
        frame_dir=frame_dir,
        image_size=image_size,
        seq_len=seq_len
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Generator / Discriminator
    G = MiniDriveGANPaperLike(  
        hidden_dim=hidden_dim,
        theme_dim=theme_dim,
        action_dim=action_dim
    ).to(device)

    D = Discriminator().to(device)

    # Optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # Losses
    recon_loss_fn = nn.L1Loss()
    theme_loss_fn = nn.MSELoss()
    content_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCELoss()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for x_seq, action, x_next in loader:
            x_seq = x_seq.to(device)         # [B, T, 3, H, W]
            action = action.to(device)       # [B, 4]
            x_next = x_next.to(device)       # [B, 3, H, W]

            batch_size_cur = x_next.size(0)

            real_labels = torch.ones(batch_size_cur, 1, device=device)
            fake_labels = torch.zeros(batch_size_cur, 1, device=device)

            # =========================================================
            # 1. Generator forward
            # =========================================================
            x_pred, theme_seq, content_seq, next_theme_pred, next_content_pred = G(x_seq, action)

            # =========================================================
            # 2. Train Discriminator
            # =========================================================
            optimizer_D.zero_grad()

            # real
            d_real = D(x_next)
            loss_d_real = adv_loss_fn(d_real, real_labels)

            # fake
            d_fake = D(x_pred.detach())
            loss_d_fake = adv_loss_fn(d_fake, fake_labels)

            loss_D = 0.5 * (loss_d_real + loss_d_fake)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optimizer_D.step()

            # =========================================================
            # 3. Train Generator
            # =========================================================
            optimizer_G.zero_grad()

            # reconstruction loss
            loss_recon = recon_loss_fn(x_pred, x_next)

            # latent target
            with torch.no_grad():
                # 입력 시퀀스에서 첫 프레임 제거 + 실제 다음 프레임 추가
                next_seq = torch.cat([x_seq[:, 1:], x_next.unsqueeze(1)], dim=1)

                next_theme_seq_true, next_content_seq_true = G.encode_sequence(next_seq)

                next_theme_true = next_theme_seq_true[:, -1]       # [B, theme_dim]
                next_content_true = next_content_seq_true[:, -1]   # [B, C, H, W]

            loss_theme = theme_loss_fn(next_theme_pred, next_theme_true)
            loss_content = content_loss_fn(next_content_pred, next_content_true)

            # adversarial loss for generator
            d_fake_for_g = D(x_pred)
            loss_G_adv = adv_loss_fn(d_fake_for_g, real_labels)

            # final generator loss
            loss_G = (          
                0.5 *loss_recon
                + 0.2 * loss_theme
                + 0.2 * loss_content
                + 0.01 * loss_G_adv
            )

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            optimizer_G.step()

            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()

        avg_g_loss = total_g_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}"
        )

        # =========================================================
        # 4. Save sample outputs
        # =========================================================
        G.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            x_seq, action, x_next = sample_batch

            x_seq = x_seq.to(device)
            action = action.to(device)
            x_next = x_next.to(device)

            x_pred, _, _, _, _ = G(x_seq, action)

            save_prediction_samples(
                x_seq.cpu(),
                x_next.cpu(),
                x_pred.cpu(),
                "outputs",
                epoch
            )

        # =========================================================
        # 5. Save checkpoints
        # =========================================================
        torch.save(
            G.state_dict(),
            f"checkpoints/generator_epoch_{epoch}.pth"
        )
        torch.save(
            D.state_dict(),
            f"checkpoints/discriminator_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    train()
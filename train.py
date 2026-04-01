import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DrivingSequenceDataset
from model import MiniDriveGANPaperLike
from utils import save_prediction_samples


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/labels.csv"
    frame_dir = "data/frames"

    batch_size = 8
    lr = 1e-4 #학습률 0.0001로 수정
    epochs = 20
    image_size = 128
    hidden_dim = 256
    theme_dim = 64
    seq_len = 4

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

    model = MiniDriveGANPaperLike(
        hidden_dim=hidden_dim,
        theme_dim=theme_dim,
        action_dim=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    recon_loss_fn = nn.L1Loss()
    theme_loss_fn = nn.MSELoss()
    content_loss_fn = nn.MSELoss()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_seq, action, x_next in loader:
            x_seq = x_seq.to(device)         # [B, T, 3, H, W]
            # steering = steering.to(device)   # [B, 1]
            action = action.to(device)     # [B, 4]
            x_next = x_next.to(device)       # [B, 3, H, W]

            # forward
            x_pred, theme_seq, content_seq, next_theme_pred, next_content_pred = model(x_seq, action)

            # 1. reconstruction loss
            loss_recon = recon_loss_fn(x_pred, x_next)

            # 2. latent consistency target
            with torch.no_grad():
                # 입력 시퀀스에서 첫 프레임 제거 + 실제 다음 프레임 추가
                next_seq = torch.cat([x_seq[:, 1:], x_next.unsqueeze(1)], dim=1)

                next_theme_seq_true, next_content_seq_true = model.encode_sequence(next_seq)

                # 마지막 프레임에 해당하는 latent를 정답 target으로 사용
                next_theme_true = next_theme_seq_true[:, -1]       # [B, theme_dim]
                next_content_true = next_content_seq_true[:, -1]   # [B, C, H, W]

            loss_theme = theme_loss_fn(next_theme_pred, next_theme_true)
            loss_content = content_loss_fn(next_content_pred, next_content_true)

            # 최종 loss
            loss = loss_recon + 0.1 * loss_theme + 0.1 * loss_content

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            # clip_grad_norm_으로 모델의 모든 매개변수의 그래디언트가 max_norm을 넘지 않도록 클리핑
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

        # 샘플 저장
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            x_seq, action, x_next = sample_batch

            x_seq = x_seq.to(device)
            action = action.to(device)
            x_next = x_next.to(device)

            x_pred, _, _, _, _ = model(x_seq, action)

            save_prediction_samples(
                x_seq.cpu(),
                x_next.cpu(),
                x_pred.cpu(),
                "outputs",
                epoch
            )

        torch.save(
            model.state_dict(),
            f"checkpoints/mini_drivegan_paperlike_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    train()
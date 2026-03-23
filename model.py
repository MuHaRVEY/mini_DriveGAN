"""
현재 이미지 x_t
   ↓
Encoder
   ↓
현재 latent z_t
   ↓          + 현재 action a_t
   └────────────→ Transition Model
                    ↓
               예측된 다음 latent z_{t+1}
                    ↓
                 Decoder
                    ↓
            예측된 다음 이미지 x_{t+1}
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=12): # RGB 이미지 4장을 사용하도록 3에서 12로
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),   # 128 -> 64
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 64 -> 32
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 32 -> 16
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 16 -> 8
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class TransitionModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z_t, action):
        x = torch.cat([z_t, action], dim=1)
        delta = self.net(x)
        z_next_pred = z_t + delta
        return z_next_pred


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 -> 16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 -> 32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 -> 64
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 8, 8)
        x_recon = self.deconv(h)
        return x_recon


class MiniDriveGAN(nn.Module):
    def __init__(self, latent_dim=128, action_dim=1,in_channels =12):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, in_channels = in_channels)
        self.transition = TransitionModel(latent_dim=latent_dim, action_dim=action_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x_t, action):
        z_t = self.encoder(x_t)
        z_next_pred = self.transition(z_t, action)
        x_next_pred = self.decoder(z_next_pred)
        return x_next_pred, z_t, z_next_pred
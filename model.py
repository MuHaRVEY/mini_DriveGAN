"""
[Mini DriveGAN Sequence ConvLSTM 구조]

입력:
    최근 4개의 연속된 주행 이미지 시퀀스
    x_seq = [x_{t-3}, x_{t-2}, x_{t-1}, x_t]
    shape: [B, T, 3, H, W]

처리 흐름:

    각 프레임을 Encoder(CNN)에 통과시켜 feature map으로 변환
        x_{t-3} -> z_{t-3}
        x_{t-2} -> z_{t-2}
        x_{t-1} -> z_{t-1}
        x_t     -> z_t

    이 feature map sequence를 시간 순서대로 ConvLSTM에 입력
        z_{t-3} -> ConvLSTM
        z_{t-2} -> ConvLSTM
        z_{t-1} -> ConvLSTM
        z_t     -> ConvLSTM

    마지막 hidden state에 steering action 반영
        z_{t+1} = h_t + action_embedding(a_t)

    Decoder를 통해 다음 프레임 예측
        z_{t+1} -> x_{t+1}

목표:
    현재 시퀀스와 action을 바탕으로 다음 프레임 예측
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),   # 128 -> 64
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),          # 64 -> 32
            nn.ReLU(inplace=True),

            nn.Conv2d(128, hidden_dim, 4, 2, 1),  # 32 -> 16
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        return self.conv(x)  # [B, hidden_dim, 16, 16]


class Decoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),         # 32 -> 64
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),           # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(z)


class SequenceConvLSTMTransition(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convlstm = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)

    def init_state(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c

    def forward(self, z_seq, action):
        """
        z_seq: [B, T, C, H, W]
        action: [B, 1]
        """
        bsz, seq_len, channels, height, width = z_seq.shape
        device = z_seq.device

        h, c = self.init_state(bsz, height, width, device)

        # 프레임 순서대로 ConvLSTM 처리
        for t in range(seq_len):
            z_t = z_seq[:, t]  # [B, C, H, W]
            h, c = self.convlstm(z_t, h, c)

        # 마지막 hidden state에 action 반영
        a = self.action_embed(action)              # [B, C]
        a = a.view(bsz, channels, 1, 1).expand(-1, -1, height, width)

        z_next_pred = h + a
        return z_next_pred, h, c


class MiniDriveGANSequence(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(in_channels=3, hidden_dim=hidden_dim)
        self.transition = SequenceConvLSTMTransition(hidden_dim=hidden_dim, action_dim=action_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim)

    def encode_sequence(self, x_seq):
        """
        x_seq: [B, T, 3, H, W]
        return: [B, T, C, H, W]
        """
        bsz, seq_len, ch, h, w = x_seq.shape
        encoded = []

        for t in range(seq_len):
            feat = self.encoder(x_seq[:, t])  # [B, C, 16, 16]
            encoded.append(feat)

        z_seq = torch.stack(encoded, dim=1)
        return z_seq

    def forward(self, x_seq, action):
        z_seq = self.encode_sequence(x_seq)
        z_next_pred, h, c = self.transition(z_seq, action)
        x_next_pred = self.decoder(z_next_pred)
        return x_next_pred, z_seq, z_next_pred, h, c
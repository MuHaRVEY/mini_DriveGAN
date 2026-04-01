import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
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
    def __init__(self, in_channels=3, hidden_dim=256, theme_dim=64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, hidden_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.theme_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, theme_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        theme = self.theme_head(feat)
        content = feat
        return theme, content


class ThemeTransition(nn.Module):
    def __init__(self, theme_dim=64, action_dim=4):
        super().__init__()
        self.gru = nn.GRUCell(theme_dim + action_dim, theme_dim)

    def forward(self, theme_seq, action):
        bsz, seq_len, theme_dim = theme_seq.shape
        h = theme_seq[:, 0]

        for t in range(seq_len):
            x = torch.cat([theme_seq[:, t], action], dim=1)
            h = self.gru(x, h)

        return h


class ContentTransition(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=4):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.dep_dim = hidden_dim // 2
        self.ind_dim = hidden_dim // 2

        self.dep_lstm = ConvLSTMCell(self.dep_dim, self.dep_dim)
        self.ind_lstm = ConvLSTMCell(self.ind_dim, self.ind_dim)
        self.action_embed = nn.Linear(action_dim, self.dep_dim)

    def init_state(self, batch_size, channels, height, width, device):
        h = torch.zeros(batch_size, channels, height, width, device=device)
        c = torch.zeros(batch_size, channels, height, width, device=device)
        return h, c

    def forward(self, content_seq, action):
        bsz, seq_len, channels, height, width = content_seq.shape
        device = content_seq.device

        dep_h, dep_c = self.init_state(bsz, self.dep_dim, height, width, device)
        ind_h, ind_c = self.init_state(bsz, self.ind_dim, height, width, device)

        a = self.action_embed(action)
        a = a.view(bsz, self.dep_dim, 1, 1).expand(-1, -1, height, width)

        for t in range(seq_len):
            feat_t = content_seq[:, t]
            dep_t, ind_t = torch.split(feat_t, [self.dep_dim, self.ind_dim], dim=1)

            dep_input = dep_t + a
            dep_h, dep_c = self.dep_lstm(dep_input, dep_h, dep_c)
            ind_h, ind_c = self.ind_lstm(ind_t, ind_h, ind_c)

        next_content = torch.cat([dep_h, ind_h], dim=1)
        return next_content


class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, theme_dim=64):
        super().__init__()
        self.theme_to_scale = nn.Linear(theme_dim, hidden_dim)
        self.theme_to_shift = nn.Linear(theme_dim, hidden_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, theme, content):
        bsz, channels, height, width = content.shape
        scale = self.theme_to_scale(theme).view(bsz, channels, 1, 1)
        shift = self.theme_to_shift(theme).view(bsz, channels, 1, 1)
        fused = content * (1 + scale) + shift
        return self.deconv(fused)


class MiniDriveGANPaperLike(nn.Module):
    def __init__(self, hidden_dim=256, theme_dim=64, action_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels=3, hidden_dim=hidden_dim, theme_dim=theme_dim)
        self.theme_transition = ThemeTransition(theme_dim=theme_dim, action_dim=action_dim)
        self.content_transition = ContentTransition(hidden_dim=hidden_dim, action_dim=action_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, theme_dim=theme_dim)

    def encode_sequence(self, x_seq):
        theme_list = []
        content_list = []

        for t in range(x_seq.size(1)):
            theme_t, content_t = self.encoder(x_seq[:, t])
            theme_list.append(theme_t)
            content_list.append(content_t)

        theme_seq = torch.stack(theme_list, dim=1)
        content_seq = torch.stack(content_list, dim=1)
        return theme_seq, content_seq

    def forward(self, x_seq, action):
        theme_seq, content_seq = self.encode_sequence(x_seq)
        next_theme = self.theme_transition(theme_seq, action)
        next_content = self.content_transition(content_seq, action)
        x_next_pred = self.decoder(next_theme, next_content)
        return x_next_pred, theme_seq, content_seq, next_theme, next_content


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)
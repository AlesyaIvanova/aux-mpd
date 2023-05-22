import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from init import SPEC_NORM_DISC, SPEC_NORM_GEN, L_CONSTANT, SUR_CONSTANT, SUR_METRIC, device
from evaluation import get_val_metric


def custom_conv(in_channels, out_channels, kernel_size, padding, is_spectral_norm=False):
    if is_spectral_norm:
        return spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        )
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)


def custom_linear(in_features, out_features, is_spectral_norm=False):
    if is_spectral_norm:
        return spectral_norm(nn.Linear(in_features, out_features))
    else:
        return nn.Linear(in_features, out_features)


def new_block(in_channels, out_channels, kernel_size=3, padding=1, p=0.2):
    return nn.Sequential(
        custom_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                    is_spectral_norm=SPEC_NORM_DISC),
        nn.ELU(),
        nn.Dropout(p=p)
    )


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(  # bs  f   h  w
            new_block(in_channels=6, out_channels=16),  # -> [bs, 16, 8, 16]
            new_block(in_channels=16, out_channels=16)  # -> [bs, 16, 8, 16]
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # -> [bs, 16, 8, 8]
            new_block(in_channels=16, out_channels=32),  # -> [bs, 32, 8, 8]
        )

        self.encoder3 = nn.Sequential(
            new_block(in_channels=32, out_channels=32),  # -> [bs, 32, 8, 8]
        )

        self.surrogate_encoder = nn.Sequential(
            new_block(in_channels=32, out_channels=32, kernel_size=5, padding=0),  # -> [bs, 32, 4, 4]
            new_block(in_channels=32, out_channels=32, kernel_size=4, padding=0),  # -> [bs, 32, 1, 1]
        )

        self.surrogate_linear = nn.Sequential(
            nn.Linear(in_features=32, out_features=1),  # -> [bs, 1, 1, 1]
        )
        
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # -> [bs, 32, 4, 4]
            new_block(in_channels=32, out_channels=64, padding=0),  # -> [bs, 64, 2, 2]
            new_block(in_channels=64, out_channels=64, kernel_size=2, padding=0)  # -> [bs, 64, 1, 1]
        )

        self.linear = nn.Sequential(
            custom_linear(in_features=70, out_features=128, is_spectral_norm=SPEC_NORM_DISC),
            nn.ELU(),
            custom_linear(in_features=128, out_features=1, is_spectral_norm=SPEC_NORM_DISC)
        )

    def forward(self, features, image):  # features [bs, 5]  image [bs, 8, 16]
        bs, h, w = image.shape
        _, k = features.shape
        in_image = image.reshape((bs, 1, h, w))
        in_features = features.reshape((bs, k, 1, 1))
        in_features = torch.tile(in_features, (1, 1, h, w))  # -> features [bs, k, h, w]
        out = torch.cat((in_features, in_image), dim=1)  # -> [bs, k+1, h, w]

        out = self.encoder1(out)
        out = F.pad(out, (0, 0, h // 2, h // 2))
        out = self.encoder2(out)

        out_sur = self.surrogate_encoder(out)
        out_sur = out_sur.squeeze()
        out_sur = self.surrogate_linear(out_sur)

        out = self.encoder3(out)

        out = self.encoder4(out)
        out = out.squeeze()
        
        out = torch.cat((features, out, out_sur), dim=1)  # -> [bs, 70]
        out = self.linear(out)

        out = L_CONSTANT * out
        return out, out_sur


class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shift = 0.01
        val = np.log10(2)
        v0 = np.log10(2) / 10
        return torch.where(
            x > shift,
            val + x - shift,
            v0 + nn.ELU(alpha=(v0 * shift / (val - v0)))(x) * (val - v0) / shift
        )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            custom_linear(in_features=37, out_features=32, is_spectral_norm=SPEC_NORM_GEN),
            nn.ELU(),
            custom_linear(in_features=32, out_features=64, is_spectral_norm=SPEC_NORM_GEN),
            nn.ELU(),
            custom_linear(in_features=64, out_features=64, is_spectral_norm=SPEC_NORM_GEN),
            nn.ELU(),
            custom_linear(in_features=64, out_features=64, is_spectral_norm=SPEC_NORM_GEN),
            nn.ELU(),
            custom_linear(in_features=64, out_features=128, is_spectral_norm=SPEC_NORM_GEN),
            CustomActivation()
        )

    def forward(self, features, image):  # features [bs, 5]  image [bs, 32]
        bs, n = image.shape
        _, k = features.shape
        out = torch.cat((features, image), dim=1)  # -> [bs, 37]
        out = self.encoder(out)  # -> [bs, 128]
        out = out.reshape((bs, 8, 16))  # -> [bs, 8, 16]
        return out

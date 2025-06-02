import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 Conv → BatchNorm → ReLU (no dropout inside).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First 3×3 conv
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second 3×3 conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Down‐sampling step: MaxPool2d → Dropout2d → DoubleConv.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv    = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Up‐sampling step:
      1) ConvTranspose2d to upsample from in_channels→out_channels
      2) (pad/resize if needed) so that the upsampled matches the skip feature’s spatial dims
      3) Concatenate the skip feature (skip_channels) with the upsampled (out_channels)
      4) DoubleConv on (skip_channels + out_channels) → out_channels
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1) Upsample from in_channels → out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 2) After concatenation, total channels = skip_channels + out_channels
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # upsample
        # If spatial dims don’t match exactly, interpolate to match x2
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # Concatenate along channel dim
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1×1 Conv → Softmax to produce pixelwise class probabilities.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.conv(x)
        return self.softmax(logits)

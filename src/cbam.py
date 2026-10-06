import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        batch_size, channels, _, _ = x.shape

        avg_features = self.avg_pool(x)

        max_features = self.max_pool(x)

        avg_features = avg_features.view(
            batch_size,
            channels
        )

        max_features = max_features.view(
            batch_size,
            channels
        )

        avg_attention = self.mlp(avg_features)

        max_attention = self.mlp(max_features)

        attention = avg_attention + max_attention

        attention = self.sigmoid(attention)

        attention = attention.view(
            batch_size,
            channels,
            1,
            1
        )

        return x * attention
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_features = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_features = torch.max(
            x,
            dim=1,
            keepdim=True
        )[0]

        combined = torch.cat(
            [avg_features, max_features],
            dim=1
        )

        attention = self.conv(combined)

        attention = self.sigmoid(attention)

        return x * attention
class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


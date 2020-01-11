import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels_list):
        super(UNet, self).__init__()

        self.in_conv = DoubleConv(in_channels, down_channels_list[0])

        self.downs = torch.nn.ModuleList()
        cur_channels = down_channels_list[0]
        for channels in down_channels_list[1:]:
            self.downs.append(Down(cur_channels, channels))
            cur_channels = channels

        self.ups = torch.nn.ModuleList()
        for channels in reversed(down_channels_list[:-1]):
            self.ups.append(Up(cur_channels + channels, channels))
            cur_channels = channels

        self.out_conv = nn.Conv1d(down_channels_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        intermediates = []
        for down in self.downs:
            intermediates.append(x)
            x = down(x)
        for up in self.ups:
            x = up(x, intermediates.pop())
        x = self.out_conv(x)
        return x

def Conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def DoubleConv(in_channels, out_channels):
    return nn.Sequential(
        Conv(in_channels, out_channels), Conv(out_channels, out_channels)
    )


def Down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2), DoubleConv(in_channels, out_channels)
    )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, low_res, high_res):
        low_res = self.up_sample(low_res)
        concatenation = torch.cat([high_res, low_res], dim=1)
        return self.conv(concatenation)

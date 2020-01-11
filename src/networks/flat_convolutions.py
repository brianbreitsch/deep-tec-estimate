import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlatConvolutions(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_sizes_list):
        super(FlatConvolutions, self).__init__()
        layers = []
        for i in range(len(kernel_sizes_list)):
            layers.append(torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=out_channels_list[i-1] if i > 0 else in_channels,
                    out_channels=out_channels_list[i],
                    kernel_size=kernel_sizes_list[i],
                    padding=padding_for_kernel_size(kernel_sizes_list[i])
                ),
                torch.nn.BatchNorm1d(out_channels_list[i]),
                torch.nn.LeakyReLU()
            ))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def padding_for_kernel_size(kernel_size):
    return kernel_size // 2

import os

import fire
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ScintillationDataset
from .networks import FlatConvolutions, UNet


def train(num_epochs=100, logdir="/tmp/deep-tec-estimate"):
    network = FlatConvolutions(
        in_channels=6, out_channels_list=[1], kernel_sizes_list=[5]
    )
    train_dataset = ScintillationDataset(1000, 0)
    val_dataset = ScintillationDataset(1000, 0)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=1)
    optimizer = torch.optim.Adam(lr=1e-4, params=network.parameters())
    os.makedirs(logdir, exist_ok=True)
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader)
        for x, y in pbar:
            optimizer.zero_grad()
            yhat = network(x).squeeze(dim=1)
            loss = torch.nn.functional.mse_loss(yhat, y)
            pbar.set_description(f"epoch: {epoch} training mse: {loss}.2f")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            for batch_idx, (x, y) in enumerate(pbar):
                yhat = network(x).squeeze(dim=1)
                loss = torch.nn.functional.mse_loss(yhat, y)
                pbar.set_description(f"epoch: {epoch} validation mse: {loss}.2f")

                if batch_idx == 0:
                    for i in range(min(10, yhat.shape[0])):
                        sample_plot = make_sample_plot(x[i], y[i], yhat[i])
                        imageio.imwrite(
                            os.path.join(logdir, f"plot_{i}.jpg"), sample_plot
                        )


def make_sample_plot(_x, _y, _y_hat):
    x, y, y_hat = (
        _x.detach().cpu().numpy(),
        _y.detach().cpu().numpy(),
        _y_hat.detach().cpu().numpy(),
    )
    fig = plt.figure(figsize=(10, 3), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(y, label="label")
    ax.plot(y_hat, label="prediction")
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    sample_plot = (
        np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        .reshape(height, width, 3)
        .astype(np.uint8)
    )
    plt.close(fig)
    return sample_plot


if __name__ == "__main__":
    fire.Fire(train)

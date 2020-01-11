
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy
import imageio
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .dataset import ScintillationDataset

class Module(pl.LightningModule):

    def __init__(self, network):
        super(Module, self).__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if batch_idx == 0:
            for i in range(min(10, y_hat.shape[0])):
                sample_plot = make_sample_plot(x[i], y[i], y_hat[i])
                self.logger.experiment.add_image(f'sample-plot-{i}', sample_plot, self.current_epoch)

        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), amsgrad=True)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(ScintillationDataset(10000, 0), 32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(ScintillationDataset(1000, 1), 32)

    @pl.data_loader
    def test_dataloader(self):
        pass


def make_sample_plot(_x, _y, _y_hat):
    x, y, y_hat = _x.cpu(), _y.cpu(), _y_hat.cpu()
    fig = plt.figure(figsize=(10, 3), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(y[:])
    ax.plot(y_hat.detach().numpy()[0, :])
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi() 
    width, height = int(width), int(height)
    sample_plot = numpy.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3).astype(numpy.uint8)
    sample_plot = numpy.transpose(sample_plot, (2, 0, 1))
    return sample_plot
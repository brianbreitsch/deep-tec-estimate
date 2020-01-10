from pytorch_lightning import Trainer

from ..module import Module
from ..networks.unet import UNet

network = UNet(
    in_channels=6,
    out_channels=1, 
    down_channels_list=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 2048]
)
module = Module(network)
trainer = Trainer(gpus=1)
trainer.fit(module)
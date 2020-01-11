from pytorch_lightning import Trainer

from ..module import Module
from ..networks.flat_convolutions import FlatConvolutions

network = FlatConvolutions(
    in_channels=6,
    out_channels_list=[128] * 9 + [1],
    kernel_sizes_list=[5]*10
)
module = Module(network)
trainer = Trainer(gpus=1)
trainer.fit(module)
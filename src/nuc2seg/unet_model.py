""" Full assembly of the parts to form the complete network """

from torch.nn import Embedding
import numpy as np
from nuc2seg.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class SparseUNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_shape, n_filters=10, bilinear=False):
        super(SparseUNet, self).__init__()
        self.img_shape = tuple(img_shape) + (n_filters,)
        self.filters = Embedding(n_channels, n_filters)
        self.unet = UNet(n_filters, n_classes, bilinear=bilinear)
        self.n_classes = n_classes

    def forward(self, x, y, z):
        mask = z > -1
        b = torch.as_tensor(
            np.tile(np.arange(z.shape[0]), (z.shape[1], 1)).T[mask.numpy().astype(bool)]
        )
        W = self.filters(z[mask])
        t_input = torch.Tensor(np.zeros((z.shape[0],) + self.img_shape))
        t_input.index_put_(
            (b, torch.LongTensor(x[mask]), torch.LongTensor(y[mask])),
            W,
            accumulate=True,
        )
        t_input = torch.Tensor.permute(
            t_input, (0, 3, 1, 2)
        )  # Needs to be Batch x Channels x ImageX x ImageY
        return torch.Tensor.permute(
            self.unet(t_input), (0, 2, 3, 1)
        )  # Map back to Batch x ImageX x Image Y x Classes

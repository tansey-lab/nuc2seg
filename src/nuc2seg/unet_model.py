""" Full assembly of the parts to form the complete network """

from torch.nn import Embedding
import numpy as np
from nuc2seg.unet_parts import *
from pytorch_lightning.core import LightningModule
from nuc2seg.data import TiledDataset, Nuc2SegDataset
from torch import optim
from nuc2seg.evaluate import dice_coeff


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


def angle_loss(predictions, targets):
    """Angles are expressed in [0,1] but we want 0.01 and 0.99 to be close.
    So we take the minimum of the losses between the original prediction,
    adding 1, and subtracting 1 such that we consider 0.01, 1.01, and -1.01.
    That way 0.01 and 0.99 are only 0.02 apart."""
    delta = torch.sigmoid(predictions) - targets
    return torch.minimum(torch.minimum(delta**2, (delta - 1) ** 2), (delta + 1) ** 2)


class SparseUNet(LightningModule):
    def __init__(
        self,
        dataset: Nuc2SegDataset,
        n_channels,
        tile_height=64,
        tile_width=64,
        tile_overlap=0.25,
        n_filters=10,
        bilinear=False,
        lr: float = 1e-5,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.img_shape = (
            tile_width,
            tile_height,
            n_filters,
        )
        self.filters = Embedding(n_channels, n_filters)
        self.n_classes = dataset.n_classes + 2
        self.unet = UNet(
            self.hparams.n_filters, self.n_classes, bilinear=self.hparams.bilinear
        )
        self.foreground_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.dataset = dataset
        self.tiled_dataset = TiledDataset(
            dataset,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_overlap=tile_overlap,
        )
        # Class imbalance reweighting
        self.celltype_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=torch.Tensor(
                self.tiled_dataset.per_tile_class_histograms[:, 2:].mean()
                / self.tiled_dataset.per_tile_class_histograms[:, 2:].mean(axis=0)
            ),
        )

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

    def configure_optimizers(self):
        return optim.RMSprop(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.momentum,
        )

    def training_step(self, batch, batch_idx):
        x, y, z, labels, angles, classes, label_mask, nucleus_mask = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["angles"],
            batch["classes"],
            batch["label_mask"],
            batch["nucleus_mask"],
        )

        label_mask = label_mask.type(torch.bool)

        mask_pred = self.forward(x, y, z)

        foreground_pred = mask_pred[..., 0]
        angles_pred = mask_pred[..., 1]
        class_pred = mask_pred[..., 2:]

        # Add the cross-entropy loss on just foreground vs background
        loss = self.foreground_criterion(
            foreground_pred[label_mask], (labels[label_mask] > 0).type(torch.float)
        )

        # If there are any cells in this tile
        if nucleus_mask.sum() > 0:
            # Add the squared error loss on the correct angles for known class pixels
            loss += angle_loss(angles_pred[nucleus_mask], angles[nucleus_mask]).mean()

            # Add the cross-entropy loss on the cell type prediction for nucleus pixels
            loss += self.celltype_criterion(
                class_pred[nucleus_mask], classes[nucleus_mask] - 1
            )

        return loss

    def validation_step(self, batch, batch_idx):
        dice_score = 0

        x, y, z, labels, label_mask = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["label_mask"],
        )
        batch_size = x.shape[0]

        label_mask = label_mask.type(torch.bool)
        mask_true = (labels > 0).type(torch.float)

        # predict the mask
        mask_pred = self.forward(x, y, z)

        foreground_pred = torch.sigmoid(mask_pred[..., 0])

        for im_pred, im_true, im_label_mask in zip(
            foreground_pred, mask_true, label_mask
        ):
            im_pred, im_true = im_pred[im_label_mask], im_true[im_label_mask]

            im_pred = (im_pred > 0.5).float()
            # compute the Dice score
            dice_score += (
                dice_coeff(im_pred, im_true, reduce_batch_first=False)
                / mask_true.shape[0]
            )

        return dice_score / batch_size

""" Full assembly of the parts to form the complete network """

from torch.nn import Embedding
from nuc2seg.unet_parts import *
from pytorch_lightning.core import LightningModule, LightningDataModule
from nuc2seg.data import TiledDataset, Nuc2SegDataset
from torch import optim
from nuc2seg.evaluate import (
    dice_coeff,
    foreground_accuracy,
    squared_angle_difference,
    angle_accuracy,
)
from torch.utils.data import DataLoader, random_split


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
    return squared_angle_difference(torch.sigmoid(predictions), targets)


class SparseUNet(LightningModule):
    def __init__(
        self,
        n_channels,
        n_classes,
        celltype_criterion_weights,
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
        self.save_hyperparameters()
        self.img_shape = (
            tile_width,
            tile_height,
            n_filters,
        )
        self.filters = Embedding(n_channels, n_filters)
        self.n_classes = n_classes + 2
        self.unet = UNet(
            self.hparams.n_filters, self.n_classes, bilinear=self.hparams.bilinear
        )
        self.foreground_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # Class imbalance reweighting
        self.celltype_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=celltype_criterion_weights,
        )
        self.validation_step_outputs = []

    def forward(self, x, y, z):
        mask = z > -1
        b = (
            torch.tile(torch.arange(z.shape[0]), (z.shape[1], 1))
            .to(self.device)
            .T[mask]
        )
        W = self.filters(z[mask])
        t_input = torch.zeros((z.shape[0],) + self.img_shape, device=self.device)
        t_input.index_put_(
            (b, x[mask], y[mask]),
            W,
            accumulate=True,
        )
        t_input = torch.permute(
            t_input, (0, 3, 1, 2)
        )  # Needs to be Batch x Channels x ImageX x ImageY
        return torch.permute(
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
        foreground_loss = self.foreground_criterion(
            foreground_pred[label_mask], (labels[label_mask] > 0).type(torch.float)
        )

        self.log("foreground_loss", foreground_loss)

        # If there are any cells in this tile
        if nucleus_mask.sum() > 0:
            # Add the squared error loss on the correct angles for known class pixels
            angle_loss_val = angle_loss(
                angles_pred[nucleus_mask], angles[nucleus_mask]
            ).mean()

            self.log("angle_loss", angle_loss_val)

            # Add the cross-entropy loss on the cell type prediction for nucleus pixels
            celltype_loss = self.celltype_criterion(
                class_pred[nucleus_mask], classes[nucleus_mask] - 1
            )
            self.log("celltype_loss", celltype_loss)

            train_loss = foreground_loss + angle_loss_val + celltype_loss
        else:
            train_loss = foreground_loss

        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y, z, labels, angles = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["angles"],
        )
        # predict the mask
        prediction = self.forward(x, y, z)
        foreground_accuracy_value = foreground_accuracy(prediction, labels)
        angle_accuracy_value = angle_accuracy(
            predictions=prediction,
            target=angles,
            labels=labels,
        )

        self.validation_step_outputs.append(
            {
                "foreground_accuracy": foreground_accuracy_value,
                "angle_accuracy": angle_accuracy_value,
            }
        )

    def on_validation_epoch_end(self):
        foreground_accuracy_value = torch.stack(
            [x["foreground_accuracy"] for x in self.validation_step_outputs]
        ).mean()
        angle_accuracy_value = torch.stack(
            [x["angle_accuracy"] for x in self.validation_step_outputs]
        ).mean()
        self.log("foreground_accuracy", foreground_accuracy_value)
        self.log("angle_accuracy", angle_accuracy_value)
        self.log("val_accuracy", (foreground_accuracy_value + angle_accuracy_value) / 2)
        self.validation_step_outputs.clear()


class Nuc2SegDataModule(LightningDataModule):
    def __init__(
        self,
        preprocessed_data_path: str,
        val_percent: float = 0.1,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        tile_height: int = 64,
        tile_width: int = 64,
        tile_overlap: float = 0.25,
        num_workers: int = 0,
    ):
        super().__init__()
        self.preprocessed_data_path = preprocessed_data_path
        self.val_percent = val_percent
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_overlap = tile_overlap
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):

        self.dataset = Nuc2SegDataset.load_h5(self.preprocessed_data_path)

        dataset = TiledDataset(
            self.dataset,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            tile_overlap=self.tile_overlap,
        )
        n_val = int(len(dataset) * self.val_percent)
        n_train = len(dataset) - n_val
        self.train_set, self.val_set = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        if self.dataset is None:
            raise ValueError("You must call setup() before train_dataloader()")

        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.dataset is None:
            raise ValueError("You must call setup() before train_dataloader()")

        return DataLoader(
            self.val_set, batch_size=self.val_batch_size, num_workers=self.num_workers
        )

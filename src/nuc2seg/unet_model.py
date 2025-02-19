""" Full assembly of the parts to form the complete network """

from typing import Optional

from pytorch_lightning.core import LightningModule, LightningDataModule
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset

from nuc2seg.data import TiledDataset, Nuc2SegDataset, collate_tiles
from nuc2seg.evaluate import (
    foreground_accuracy,
    squared_angle_difference,
    angle_accuracy,
    celltype_accuracy,
)
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


def calculate_angle_loss(predictions, targets):
    return squared_angle_difference(torch.sigmoid(predictions), targets)


def calculate_even_weights(values):
    # number of non nan values
    values_non_nan = [v for v in values if not v.isnan()]

    n_classes = max(float(len(values_non_nan)), 1)

    if len(values_non_nan) == 0:
        return tuple([1.0] * len(values))

    total = sum(values_non_nan)

    result = []
    for val in values:
        result.append(total / (val * n_classes))
    return tuple(result)


def calculate_unlabeled_foreground_loss(
    x,
    y,
    gene,
    label_mask,
    foreground_pred,
    class_pred,
    background_frequencies,
    celltype_frequencies,
):
    """
    :param x: vector of transcript x coordinates, shape (batch_dim, n_transcripts,)
    :param y: vector of transcript y coordinates, shape (batch_dim, n_transcripts,)
    :param gene: vector of gene indices, shape (batch_dim, n_transcripts,)
    :param label_mask: mask of pixels, True if labeled as foreground/background, false if unlabeled, shape (batch_dim, tile_height, tile_width)
    :param foreground_pred: foreground/background prediction, shape (batch_dim, tile_height, tile_width)
    :param class_pred: cell type prediction, shape (batch_dim, tile_height, tile_width, n_classes)
    :param background_frequencies: prior frequency of each gene in the background, shape (n_genes,)
    :param celltype_frequencies: prior frequency of each gene in each cell type, shape (n_celltypes, n_genes)
    """
    batch_dim = foreground_pred.shape[0]
    max_n_transcripts_in_minibatch = gene.shape[1]
    ragged_mask = torch.flatten(gene) > -1
    valid_x = torch.flatten(x)[ragged_mask]
    valid_y = torch.flatten(y)[ragged_mask]
    valid_gene = torch.flatten(gene)[ragged_mask]
    batch_index = torch.arange(batch_dim).repeat_interleave(
        max_n_transcripts_in_minibatch
    )[ragged_mask]
    selection_vector = ~label_mask[batch_index, valid_x, valid_y]

    if torch.count_nonzero(selection_vector) == 0:
        return None

    unlabeled_pixels_gene = valid_gene[selection_vector]
    unlabeled_pixels_x = valid_x[selection_vector]
    unlabeled_pixels_y = valid_y[selection_vector]
    unlabeled_batch_index = batch_index[selection_vector]

    flattened_p_foreground = foreground_pred[
        unlabeled_batch_index, unlabeled_pixels_x, unlabeled_pixels_y
    ]
    flattened_p_background = (
        1
        - foreground_pred[unlabeled_batch_index, unlabeled_pixels_x, unlabeled_pixels_y]
    )
    p_celltype = class_pred[
        unlabeled_batch_index, unlabeled_pixels_x, unlabeled_pixels_y, :
    ]  # <N Batch x N Transcript x N Celltype>

    p_gene_given_background = torch.index_select(
        background_frequencies, 0, unlabeled_pixels_gene
    )  # <N Transcript>

    p_gene_given_celltype = torch.index_select(
        celltype_frequencies.T, 0, unlabeled_pixels_gene
    )  # <N Transcript x N Celltype>

    per_transcript_likelihood = (
        flattened_p_foreground * (p_gene_given_celltype * p_celltype).sum(dim=1)
    ) + (
        flattened_p_background * p_gene_given_background
    )  # <N Transcript>

    return -1 * torch.log(per_transcript_likelihood).sum()


def training_step(
    x,
    y,
    gene,
    labels,
    angles,
    classes,
    label_mask,
    nucleus_mask,
    prediction,
    foreground_criterion,
    celltype_criterion,
    background_frequencies,
    celltype_frequencies,
):
    """
    :param x: vector of transcript x coordinates, shape (n_transcripts,)
    :param y: vector of transcript y coordinates, shape (n_transcripts,)
    :param gene: vector of gene indices, shape (n_transcripts,)
    :param labels: ground truth label mask, shape (tile_height, tile_width)
    :param angles: ground truth angle mask, shape (tile_height, tile_width)
    :param classes: ground truth class mask, shape (tile_height, tile_width)
    :param label_mask: mask of pixels that are definitively labeled foreground or background
    :param nucleus_mask: mask of pixels that are definitively part of a nucleus
    :param prediction: model prediction, shape (tile_height, tile_width, n_classes + 2)
    :param foreground_criterion: loss function for foreground/background prediction
    :param celltype_criterion: loss function for cell type prediction
    :param background_frequencies: frequency of each gene in the background, shape (n_genes,)
    :param celltype_frequencies: frequency of each gene in each cell type, shape (n_celltypes, n_genes)
    """
    label_mask = label_mask.type(torch.bool)

    foreground_pred = prediction[..., 0]
    angles_pred = prediction[..., 1]
    class_pred = prediction[..., 2:]

    foreground_prob = torch.sigmoid(foreground_pred)
    classes_prob = torch.softmax(class_pred, dim=-1)

    # Add the cross-entropy loss on just foreground vs background
    foreground_loss = foreground_criterion(
        foreground_pred[label_mask], (labels[label_mask] > 0).type(torch.float)
    )

    unlabeled_foreground_loss = calculate_unlabeled_foreground_loss(
        x=x,
        y=y,
        gene=gene,
        label_mask=label_mask,
        class_pred=classes_prob,
        foreground_pred=foreground_prob,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )

    # If there are any cells in this tile
    if nucleus_mask.count_nonzero() > 0:
        # Add the squared error loss on the correct angles for known class pixels
        angle_loss_val = calculate_angle_loss(
            angles_pred[nucleus_mask], angles[nucleus_mask]
        ).mean()

        # Add the cross-entropy loss on the cell type prediction for nucleus pixels
        celltype_loss = celltype_criterion(
            class_pred[nucleus_mask], classes[nucleus_mask] - 1
        )

        return foreground_loss, unlabeled_foreground_loss, angle_loss_val, celltype_loss
    else:
        return foreground_loss, unlabeled_foreground_loss, None, None


class SparseUNet(LightningModule):
    def __init__(
        self,
        n_channels,
        n_classes,
        celltype_criterion_weights,
        celltype_frequencies,
        background_frequencies,
        tile_height=64,
        tile_width=64,
        tile_overlap=0.25,
        n_filters=10,
        bilinear=False,
        lr: float = 1e-5,
        weight_decay: float = 0,
        betas: float = (0.9, 0.999),
        angle_loss_factor: float = 1.0,
        foreground_loss_factor: float = 1.0,
        unlabeled_foreground_loss_factor: float = 1.0,
        celltype_loss_factor: float = 1.0,
        moving_average_size: int = 100,
        loss_reweighting: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.img_shape = (
            tile_width,
            tile_height,
            n_filters,
        )
        self.foreground_loss_history = torch.zeros(
            (moving_average_size,), device=self.device, dtype=torch.float
        )
        self.foreground_loss_history[:] = torch.nan
        self.unlabeled_foreground_loss_history = torch.zeros(
            (moving_average_size,), device=self.device, dtype=torch.float
        )
        self.unlabeled_foreground_loss_history[:] = torch.nan
        self.angle_loss_history = torch.zeros(
            (moving_average_size,), device=self.device, dtype=torch.float
        )
        self.angle_loss_history[:] = torch.nan
        self.celltype_loss_history = torch.zeros(
            (moving_average_size,), device=self.device, dtype=torch.float
        )
        self.celltype_loss_history[:] = torch.nan
        self.filters = XavierInitEmbedding(n_channels, n_filters)
        self.n_classes = n_classes + 2
        self.unet = UNet(
            self.hparams.n_filters, self.n_classes, bilinear=self.hparams.bilinear
        )
        self.foreground_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.validation_step_outputs = []

        self.celltype_frequencies = self.hparams.celltype_frequencies.to(self.device)
        self.background_frequencies = self.hparams.background_frequencies.to(
            self.device
        )
        self.celltype_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=torch.tensor(
                self.hparams.celltype_criterion_weights, dtype=torch.float
            ).to(self.device),
        )

    def setup(self, *args, **kwargs):
        # Make sure these hyper parameter tensors are on the correct device
        self.celltype_frequencies = self.hparams.celltype_frequencies.to(self.device)
        self.background_frequencies = self.hparams.background_frequencies.to(
            self.device
        )
        self.celltype_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=torch.tensor(
                self.hparams.celltype_criterion_weights, dtype=torch.float
            ).to(self.device),
        )

    def update_moving_average(
        self, foreground_loss, unlabeled_foreground_loss, angle_loss, celltype_loss
    ):
        if foreground_loss is not None:
            self.foreground_loss_history = torch.roll(self.foreground_loss_history, 1)
            self.foreground_loss_history[0] = foreground_loss
        if unlabeled_foreground_loss is not None:
            self.unlabeled_foreground_loss_history = torch.roll(
                self.unlabeled_foreground_loss_history, 1
            )
            self.unlabeled_foreground_loss_history[0] = unlabeled_foreground_loss
        if angle_loss is not None:
            self.angle_loss_history = torch.roll(self.angle_loss_history, 1)
            self.angle_loss_history[0] = angle_loss
        if celltype_loss is not None:
            self.celltype_loss_history = torch.roll(self.celltype_loss_history, 1)
            self.celltype_loss_history[0] = celltype_loss

    def get_weighted_losses(
        self, foreground_loss, unlabeled_foreground_loss, angle_loss, celltype_loss
    ):
        avg_foreground_loss = self.foreground_loss_history.nanmean()
        avg_unlabeled_foreground_loss = self.unlabeled_foreground_loss_history.nanmean()
        avg_angle_loss = self.angle_loss_history.nanmean()
        avg_celltype_loss = self.celltype_loss_history.nanmean()

        (
            foreground_weight,
            unlabeled_foreground_loss_weight,
            angle_weight,
            celltype_weight,
        ) = calculate_even_weights(
            [
                avg_foreground_loss,
                avg_unlabeled_foreground_loss,
                avg_angle_loss,
                avg_celltype_loss,
            ]
        )

        # remove gradient from weights
        foreground_weight = foreground_weight.detach()
        unlabeled_foreground_loss_weight = unlabeled_foreground_loss_weight.detach()
        angle_weight = angle_weight.detach()
        celltype_weight = celltype_weight.detach()

        return (
            (
                foreground_loss * foreground_weight
                if foreground_loss is not None
                else None
            ),
            (
                unlabeled_foreground_loss * unlabeled_foreground_loss_weight
                if unlabeled_foreground_loss is not None
                else None
            ),
            angle_loss * angle_weight if angle_loss is not None else None,
            celltype_loss * celltype_weight if celltype_loss is not None else None,
        )

    def forward(self, x, y, z):
        """

        :param x: X coordinate of transcripts, <Batch x Max N Transcripts per Tile in Batch>
        :param y: Y coordinate of transcripts, <Batch x Max N Transcripts per Tile in Batch>
        :param z: gene id, <Batch x Max N Transcripts per Tile in Batch>
        """
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
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

    def training_step(self, batch, batch_idx):
        (x, y, gene, labels, angles, classes, label_mask, nucleus_mask) = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["angles"],
            batch["classes"],
            batch["label_mask"],
            batch["nucleus_mask"],
        )

        prediction = self.forward(x, y, gene)

        foreground_loss, unlabeled_foreground_loss, angle_loss, celltype_loss = (
            training_step(
                x=x,
                y=y,
                gene=gene,
                labels=labels,
                angles=angles,
                classes=classes,
                label_mask=label_mask,
                nucleus_mask=nucleus_mask,
                prediction=prediction,
                foreground_criterion=self.foreground_criterion,
                celltype_criterion=self.celltype_criterion,
                background_frequencies=self.background_frequencies,
                celltype_frequencies=self.celltype_frequencies,
            )
        )

        self.log("foreground_loss", foreground_loss)
        total_loss = foreground_loss

        if unlabeled_foreground_loss is not None:
            total_loss = total_loss + unlabeled_foreground_loss
            self.log("unlabeled_foreground_loss", unlabeled_foreground_loss)

        if angle_loss is not None:
            total_loss = total_loss + angle_loss
            self.log("angle_loss", angle_loss)

        if celltype_loss is not None:
            total_loss = total_loss + celltype_loss
            self.log("celltype_loss", celltype_loss)

        self.log("loss", total_loss)

        if self.hparams.loss_reweighting:
            self.update_moving_average(
                foreground_loss, unlabeled_foreground_loss, angle_loss, celltype_loss
            )

        # if greater than n steps
        if self.global_step > self.hparams.moving_average_size:
            if self.hparams.loss_reweighting:
                if (
                    angle_loss is not None
                    or celltype_loss is not None
                    or unlabeled_foreground_loss is not None
                ):
                    (
                        foreground_loss,
                        unlabeled_foreground_loss,
                        angle_loss,
                        celltype_loss,
                    ) = self.get_weighted_losses(
                        foreground_loss,
                        unlabeled_foreground_loss,
                        angle_loss,
                        celltype_loss,
                    )

        foreground_loss = foreground_loss * self.hparams.foreground_loss_factor
        self.log("foreground_loss_weighted", foreground_loss)

        train_loss = foreground_loss

        if angle_loss is not None:
            angle_loss = angle_loss * self.hparams.angle_loss_factor
            self.log("angle_loss_weighted", angle_loss)
            train_loss = train_loss + angle_loss

        if celltype_loss is not None:
            celltype_loss = celltype_loss * self.hparams.celltype_loss_factor
            self.log("celltype_loss_weighted", celltype_loss)
            train_loss = train_loss + celltype_loss

        if unlabeled_foreground_loss is not None:
            unlabeled_foreground_loss = (
                unlabeled_foreground_loss
                * self.hparams.unlabeled_foreground_loss_factor
            )
            self.log("unlabeled_foreground_loss_weighted", unlabeled_foreground_loss)
            train_loss = train_loss + unlabeled_foreground_loss

        self.log("weighted_train_loss", train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y, z, labels, angles, classes, nucleus_mask = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["angles"],
            batch["classes"],
            batch["nucleus_mask"],
        )

        if nucleus_mask.count_nonzero() == 0:
            # We can't validate anything if
            # there's no nucleus in this tile
            return

        # predict the mask
        prediction = self.forward(x, y, z)
        foreground_accuracy_value = foreground_accuracy(prediction, labels)
        angle_accuracy_value = angle_accuracy(
            predictions=prediction,
            target=angles,
            labels=labels,
        )

        celltype_accuracy_value = celltype_accuracy(prediction, classes)

        self.validation_step_outputs.append(
            {
                "foreground_accuracy": foreground_accuracy_value,
                "angle_accuracy": angle_accuracy_value,
                "celltype_accuracy": celltype_accuracy_value,
            }
        )

    def predict_step(self, batch, batch_idx):
        x, y, z = (
            batch["X"],
            batch["Y"],
            batch["gene"],
        )
        return {"value": self.forward(x, y, z), "tile_index": batch_idx}

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        foreground_accuracy_value = torch.stack(
            [x["foreground_accuracy"] for x in self.validation_step_outputs]
        ).nanmean()
        angle_accuracy_value = torch.stack(
            [x["angle_accuracy"] for x in self.validation_step_outputs]
        ).nanmean()
        celltype_accuracy_value = torch.stack(
            [x["celltype_accuracy"] for x in self.validation_step_outputs]
        ).nanmean()
        self.log("foreground_accuracy", foreground_accuracy_value)
        self.log("angle_accuracy", angle_accuracy_value)
        self.log("celltype_accuracy", celltype_accuracy_value)
        self.log(
            "val_accuracy",
            (foreground_accuracy_value + angle_accuracy_value + celltype_accuracy_value)
            / 3.0,
        )
        self.validation_step_outputs.clear()


def equal_length_splits(total_size: int, n_splits: int) -> list[list[int]]:
    """
    Create a list of roughly equal integer splits that sum to total_size.

    Args:
        total_size: Total number of items to split
        n_splits: Number of groups to split into

    Returns:
        List of integer lengths that sum to total_size

    Example:
        >>> equal_length_splits(100, 3)
        [34, 33, 33]
        >>> equal_length_splits(10, 3)
        [4, 3, 3]
    """
    base_size = total_size // n_splits
    remainder = total_size % n_splits

    # First 'remainder' groups get an extra item to distribute the remainder
    sizes = [base_size + 1 if i < remainder else base_size for i in range(n_splits)]

    idx_gen = iter(range(total_size))

    output = []

    for s in sizes:
        output.append([next(idx_gen) for _ in range(s)])
    return output


class Nuc2SegDataModule(LightningDataModule):
    def __init__(
        self,
        preprocessed_data_path: str,
        val_percent: float = 0.1,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        predict_batch_size: int = 1,
        tile_height: int = 64,
        tile_width: int = 64,
        tile_overlap: float = 0.25,
        num_workers: int = 0,
        predict_n_threads: Optional[int] = None,
        predict_thread_idx: Optional[int] = None,
    ):
        super().__init__()
        self.preprocessed_data_path = preprocessed_data_path
        self.val_percent = val_percent
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_batch_size = predict_batch_size
        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.predict_set = None
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_overlap = tile_overlap
        self.num_workers = num_workers
        self.predict_n_threads = predict_n_threads
        self.predict_thread_idx = predict_thread_idx

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
        n_val = max(int(len(dataset) * self.val_percent), 1)
        n_train = len(dataset) - n_val

        if n_val <= 0 or n_train <= 0:
            raise ValueError("Not enough data to split into train and validation sets")

        if self.predict_n_threads is not None and self.predict_thread_idx is not None:
            self.predict_set = Subset(
                dataset,
                equal_length_splits(len(dataset), self.predict_n_threads)[
                    self.predict_thread_idx
                ],
            )
        else:
            self.predict_set = dataset

        self.train_set, self.val_set = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        if self.dataset is None:
            raise ValueError("You must call setup() before train_dataloader()")

        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_tiles,
        )

    def val_dataloader(self):
        if self.dataset is None:
            raise ValueError("You must call setup() before train_dataloader()")

        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_tiles,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.predict_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_tiles,
        )

from nuc2seg.unet_model import (
    SparseUNet,
    Nuc2SegDataModule,
    training_step,
    calculate_even_weights,
    calculate_unlabeled_foreground_loss,
)
import torch
import shutil
import numpy as np
from pytorch_lightning import Trainer
from nuc2seg.data import Nuc2SegDataset
from nuc2seg.preprocessing import cart2pol
import tempfile
import os
from torch import nn


def test_model_predict():
    celltype_frequencies = torch.zeros((3, 3), dtype=torch.float)
    celltype_frequencies[:] = 0.5
    background_frequencies = torch.zeros((3,), dtype=torch.float)
    background_frequencies[:] = 1e-3

    unet = SparseUNet(
        n_channels=3,
        n_classes=3,
        tile_height=64,
        tile_width=64,
        tile_overlap=0.0,
        celltype_criterion_weights=torch.tensor([1, 1, 1]),
        celltype_frequencies=celltype_frequencies,
        background_frequencies=background_frequencies,
    )

    pred = unet.forward(
        x=torch.tensor([[0, 1, 0, 1]]),
        y=torch.tensor([[0, 0, 1, 1]]),
        z=torch.tensor([[0, 1, 2, 2]]),
    )

    assert pred.shape == (1, 64, 64, 5)


def test_model():
    tile_height = 50
    tile_width = 50
    tile_overlap = 0.0
    celltype_frequencies = torch.zeros((3, 3), dtype=torch.float)
    celltype_frequencies[:] = 0.5
    background_frequencies = torch.zeros((3,), dtype=torch.float)
    background_frequencies[:] = 1e-3

    model = SparseUNet(
        n_channels=3,
        n_classes=3,
        tile_height=tile_height,
        tile_width=tile_width,
        tile_overlap=tile_overlap,
        celltype_criterion_weights=torch.tensor([1, 1, 1]).float(),
        moving_average_size=1,
        celltype_frequencies=celltype_frequencies,
        background_frequencies=background_frequencies,
        loss_reweighting=True,
    )

    trainer = Trainer(accelerator="cpu", max_epochs=3)

    labels = np.zeros((100, 100))

    labels[0:5, 0:5] = -1
    labels[2:3, 2:3] = 1

    labels[70:75, 70:75] = -1
    labels[72:73, 72:73] = 2

    ds = Nuc2SegDataset(
        labels=labels,
        angles=np.zeros((100, 100)),
        classes=labels.copy(),
        transcripts=np.array(
            [
                [49, 49, 0],
                [50, 50, 1],
                [50, 50, 2],
                [0, 0, 0],
                [99, 99, 0],
                [99, 0, 0],
                [0, 99, 0],
                [70, 70, 1],
            ]
        ),
        bbox=np.array([100, 100, 200, 200]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    tmpdir = tempfile.mkdtemp()

    output_path = os.path.join(tmpdir, "test.h5")

    ds.save_h5(output_path)

    dm = Nuc2SegDataModule(
        preprocessed_data_path=output_path,
        val_percent=0,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_overlap=tile_overlap,
    )

    try:
        trainer.fit(model, dm)
    finally:
        shutil.rmtree(tmpdir)


def test_training_step():
    foreground_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    # Class imbalance reweighting
    celltype_criterion = nn.CrossEntropyLoss(
        reduction="mean",
        weight=torch.tensor([1.0, 1.0, 1.0]),
    )

    x = torch.tensor([0, 3, 5], dtype=torch.long)
    y = torch.tensor([0, 3, 5], dtype=torch.long)
    gene = torch.tensor([0, 0, 0], dtype=torch.long)
    background_frequencies = torch.tensor([0.01])
    celltype_frequencies = torch.tensor([[0.5], [0.5], [0.5]])

    labels = torch.zeros((10, 10))
    labels[3:8, 3:8] = -1
    labels[4:7, 4:7] = 1

    classes = torch.zeros((10, 10))
    classes[4:7, 4:7] = 3

    labels_mask = (labels > -1).bool()
    nucleus_mask = (labels > 0).bool()

    predictions = torch.zeros((10, 10, 5)).float()
    predictions[:, :, 0] = torch.logit(torch.tensor(1e-5))
    predictions[3:8, 3:8, 0] = torch.logit(torch.tensor(0.9999))

    angles = torch.zeros((10, 10))

    for x in range(10):
        for y in range(10):
            x_component = 5 - x
            y_component = 5 - y
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    angles = (angles + np.pi) / (2 * np.pi)

    predictions[:, :, 1] = torch.logit(angles)

    predictions[:, :, 2:] = torch.logit(torch.tensor([0.01, 0.01, 0.99]))
    classes = classes.long()

    (
        perfect_foreground_loss,
        _,
        perfect_angle_loss,
        perfect_celltype_loss,
    ) = training_step(
        x=x,
        y=y,
        gene=gene,
        prediction=predictions,
        foreground_criterion=foreground_criterion,
        celltype_criterion=celltype_criterion,
        labels=labels,
        classes=classes,
        label_mask=labels_mask,
        nucleus_mask=nucleus_mask,
        angles=angles,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )
    perfect_train_loss = (
        perfect_foreground_loss + perfect_angle_loss + perfect_celltype_loss
    )

    assert torch.isclose(perfect_foreground_loss, torch.tensor(0.0), atol=1e-3)
    assert torch.isclose(perfect_angle_loss, torch.tensor(0.0), atol=1e-3)
    assert torch.isclose(perfect_celltype_loss, torch.tensor(0.0), atol=1e-3)

    for x in range(10):
        for y in range(10):
            x_component = 1 - x
            y_component = 1 - y
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    angles = (angles + np.pi) / (2 * np.pi)

    bad_foreground_loss, _, bad_angle_loss, bad_celltype_loss = training_step(
        x=x,
        y=y,
        gene=gene,
        prediction=predictions,
        foreground_criterion=foreground_criterion,
        celltype_criterion=celltype_criterion,
        labels=labels,
        classes=classes,
        label_mask=labels_mask,
        nucleus_mask=nucleus_mask,
        angles=angles,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )
    train_loss_worse_1 = bad_foreground_loss + bad_angle_loss + bad_celltype_loss

    assert bad_angle_loss > perfect_angle_loss

    predictions[:, :, 2:] = torch.logit(torch.tensor([0.33, 0.33, 0.33]))

    bad_foreground_loss, _, bad_angle_loss, bad_celltype_loss = training_step(
        x=x,
        y=y,
        gene=gene,
        prediction=predictions,
        foreground_criterion=foreground_criterion,
        celltype_criterion=celltype_criterion,
        labels=labels,
        classes=classes,
        label_mask=labels_mask,
        nucleus_mask=nucleus_mask,
        angles=angles,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )
    train_loss_worse_2 = bad_foreground_loss + bad_angle_loss + bad_celltype_loss

    assert bad_celltype_loss > perfect_celltype_loss

    predictions[3:8, 3:8, 0] = torch.logit(torch.tensor(0.5))

    bad_foreground_loss, _, bad_angle_loss, bad_celltype_loss = training_step(
        x=x,
        y=y,
        gene=gene,
        prediction=predictions,
        foreground_criterion=foreground_criterion,
        celltype_criterion=celltype_criterion,
        labels=labels,
        classes=classes,
        label_mask=labels_mask,
        nucleus_mask=nucleus_mask,
        angles=angles,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )
    train_loss_worse_3 = bad_foreground_loss + bad_angle_loss + bad_celltype_loss

    assert bad_foreground_loss > perfect_foreground_loss
    assert (
        train_loss_worse_3
        > train_loss_worse_2
        > train_loss_worse_1
        > perfect_train_loss
    )


def test_calculate_even_weights():
    x, y, z = (torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))
    a, b, c = calculate_even_weights([x, y, z])

    assert (a * x).item() == (b * y).item() == (c * z).item() == 1.0

    x, y, z = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
    a, b, c = calculate_even_weights([x, y, z])

    assert (a * x).item() == (b * y).item() == (c * z).item() == 2.0

    x, y, z = (torch.tensor([1e-7]), torch.tensor([1e-2]), torch.tensor([3.0]))
    a, b, c = calculate_even_weights([x, y, z])

    assert (a * x).item() == (b * y).item() == (c * z).item()

    x, y, z = (torch.tensor([1e-7]), torch.tensor([np.nan]), torch.tensor([3.0]))
    a, b, c = calculate_even_weights([x, y, z])

    assert not a.isnan()
    assert b.isnan()
    assert not c.isnan()

    x, y, z = (torch.tensor([1e-7]), torch.tensor([np.nan]), torch.tensor([np.nan]))
    a, b, c = calculate_even_weights([x, y, z])

    assert a == 1.0
    assert b.isnan()
    assert c.isnan()


def test_calculate_unlabeled_foreground_loss():
    x = torch.tensor([0, 1], dtype=torch.long)
    y = torch.tensor([0, 0], dtype=torch.long)
    gene = torch.tensor([0, 1], dtype=torch.long)
    foreground_pred = torch.tensor([[0.99, 0.99], [0.01, 0.01]], dtype=torch.float)
    class_pred = torch.tensor(
        [[[0.99, 0.01], [0.99, 0.01]], [[0.01, 0.99], [0.01, 0.99]]], dtype=torch.float
    )
    label_mask = torch.tensor([[0, 0], [0, 0]], dtype=torch.bool)

    background_frequencies = torch.tensor([0.01, 0.1], dtype=torch.float)
    celltype_frequencies = torch.tensor([0.5, 0.6], dtype=torch.float)

    result = calculate_unlabeled_foreground_loss(
        x=x,
        y=y,
        gene=gene,
        label_mask=label_mask,
        foreground_pred=foreground_pred,
        class_pred=class_pred,
        background_frequencies=background_frequencies,
        celltype_frequencies=celltype_frequencies,
    )

    celltype_frequencies2 = torch.tensor([0.01, 0.01], dtype=torch.float)
    background_frequencies2 = torch.tensor([0.5, 0.6], dtype=torch.float)
    result2 = calculate_unlabeled_foreground_loss(
        x=x,
        y=y,
        gene=gene,
        label_mask=label_mask,
        foreground_pred=foreground_pred,
        class_pred=class_pred,
        background_frequencies=background_frequencies2,
        celltype_frequencies=celltype_frequencies2,
    )

    assert result2 > 0
    assert result > 0
    assert result2 > result

    result_none = calculate_unlabeled_foreground_loss(
        x=x,
        y=y,
        gene=gene,
        label_mask=~label_mask,
        foreground_pred=foreground_pred,
        class_pred=class_pred,
        background_frequencies=background_frequencies2,
        celltype_frequencies=celltype_frequencies2,
    )
    assert result_none is None

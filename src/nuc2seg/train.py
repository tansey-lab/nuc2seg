import torch
import torch.nn as nn
import tqdm

from nuc2seg.data import TiledDataset, collate_tiles
from torch import optim
from torch.utils.data import DataLoader, random_split

from nuc2seg.evaluate import evaluate


def angle_loss(predictions, targets):
    """Angles are expressed in [0,1] but we want 0.01 and 0.99 to be close.
    So we take the minimum of the losses between the original prediction,
    adding 1, and subtracting 1 such that we consider 0.01, 1.01, and -1.01.
    That way 0.01 and 0.99 are only 0.02 apart."""
    delta = torch.sigmoid(predictions) - targets
    return torch.minimum(torch.minimum(delta**2, (delta - 1) ** 2), (delta + 1) ** 2)


def train(
    model,
    device,
    dataset: TiledDataset,
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    max_workers: int = 1,
    validation_frequency: int = 500,
    num_dataloader_workers: int = 4,
):

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collate_tiles,
        num_workers=num_dataloader_workers,
    )  # TODO: add num_workers back; cut out to work in ipython
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    foreground_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    celltype_criterion = nn.CrossEntropyLoss(
        reduction="mean",
        weight=torch.Tensor(
            dataset.per_tile_class_histograms[:, 2:].mean()
            / dataset.per_tile_class_histograms[:, 2:].mean(axis=0)
        ),
    )  # Class imbalance reweighting

    global_step = 0
    validation_scores = []

    # 5. Begin training
    for epoch in tqdm.trange(0, epochs, position=0, desc="Epoch"):
        model.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader, position=1, desc="Batch", leave=False):
            x, y, z, labels, angles, classes, label_mask, nucleus_mask = (
                batch["X"].to(device),
                batch["Y"].to(device),
                batch["gene"].to(device),
                batch["labels"].to(device),
                batch["angles"].to(device),
                batch["classes"].to(device),
                batch["label_mask"].to(device),
                batch["nucleus_mask"].to(device),
            )

            label_mask = label_mask.type(torch.bool)

            mask_pred = model(x, y, z)

            foreground_pred = mask_pred[..., 0]
            angles_pred = mask_pred[..., 1]
            class_pred = mask_pred[..., 2:]

            # Add the cross-entropy loss on just foreground vs background
            loss = foreground_criterion(
                foreground_pred[label_mask], (labels[label_mask] > 0).type(torch.float)
            )

            # If there are any cells in this tile
            if nucleus_mask.sum() > 0:
                # Add the squared error loss on the correct angles for known class pixels
                loss += angle_loss(
                    angles_pred[nucleus_mask], angles[nucleus_mask]
                ).mean()

                # Add the cross-entropy loss on the cell type prediction for nucleus pixels
                loss += celltype_criterion(
                    class_pred[nucleus_mask], classes[nucleus_mask] - 1
                )

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Update iters
            global_step += 1
            epoch_loss += loss.item()

            # for Evaluating model performance/ convergence
            if global_step % validation_frequency == 0:
                validation_score = evaluate(model, val_loader, device, amp)
                validation_scores.append(validation_score)

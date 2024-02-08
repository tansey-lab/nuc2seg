import torch
import torch.nn as nn
from nuc2seg.data_loading import XeniumDataset, xenium_collate_fn
from torch import optim
from torch.utils.data import DataLoader, random_split

from nuc2seg.unet_model import SparseUNet
from nuc2seg.evaluate import evaluate


def angle_loss(predictions, targets):
    """Angles are expressed in [0,1] but we want 0.01 and 0.99 to be close.
    So we take the minimum of the losses between the original prediction,
    adding 1, and subtracting 1 such that we consider 0.01, 1.01, and -1.01.
    That way 0.01 and 0.99 are only 0.02 apart."""
    delta = torch.sigmoid(predictions) - targets
    return torch.minimum(torch.minimum(delta**2, (delta - 1) ** 2), (delta + 1) ** 2)


"""
device = 'cpu'
epochs = 5
batch_size = 3
learning_rate = 1e-5
val_percent = 0.1
save_checkpoint = True
amp = False
weight_decay = 1e-8
momentum = 0.999
gradient_clipping = 1.0
max_workers = 1"""


def train(
    model,
    device,
    tiles_dir: str,
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
):
    # Create the dataset
    dataset = XeniumDataset(tiles_dir)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, pin_memory=True, collate_fn=xenium_collate_fn
    )  # TODO: add num_workers back; cut out to work in ipython
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {learning_rate}
    #     Training size:   {n_train}
    #     Validation size: {n_val}
    #     Checkpoints:     {save_checkpoint}
    #     Device:          {device.type}
    #     Mixed Precision: {amp}
    # ''')

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
            dataset.class_counts[:, 2:].mean()
            / dataset.class_counts[:, 2:].mean(axis=0)
        ),
    )  # Class imbalance reweighting

    global_step = 0
    validation_scores = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
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

            # TODO: move everything to the appropriate device if GPU training
            # images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # true_masks = true_masks.to(device=device, dtype=torch.long)
            #   with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
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

            # TODO: include DICE coefficient in loss?

            print(loss)

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Update iters
            global_step += 1
            epoch_loss += loss.item()

            if global_step % 500 == 0:
                validation_score = evaluate(model, val_loader, device, amp)
                print("Previous validation scores:")
                print(validation_scores)
                print(f"Current: {validation_score:.2f}")
                validation_scores.append(validation_score)


if __name__ == "__main__":
    transcripts_dir = "data/tiles/transcripts/"
    labels_dir = "data/tiles/labels/"
    angles_dir = "data/tiles/angles/"
    classes_dir = "data/tiles/classes/"

    n_classes = 12

    # Create the model
    # Outputs:
    # Channel 0: Foreground vs background logit
    # Channel 1: Angle logit pointing to the nucleus
    # Channel 2-K+2: Class label prediction
    # TODO: first parameter should be the number of unique transcripts
    model = SparseUNet(600, n_classes + 2, (64, 64))

    device = "cpu"

    train(model, device, transcripts_dir, labels_dir, angles_dir, classes_dir)

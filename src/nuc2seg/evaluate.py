import torch
import numpy as np
from scipy.special import softmax, expit
from torch import Tensor
from matplotlib import pyplot as plt
import tqdm

from nuc2seg.xenium_utils import pol2cart


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    inter = 2 * (input * target).sum()
    sets_sum = input.sum() + target.sum()
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum) # what is this doing???

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# @torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)  # TODO: multiply by batch_size right????
    dice_score = 0

    # iterate over the validation set
    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
    for idx, batch in enumerate(
        tqdm.tqdm(dataloader, desc="Validation", unit="batch", position=3)
    ):
        x, y, z, labels, label_mask = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"],
            batch["label_mask"],
        )
        label_mask = label_mask.type(torch.bool)
        mask_true = (labels > 0).type(torch.float)

        # move images and labels to correct device and type
        # image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        # mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        mask_pred = net(x, y, z)

        if mask_pred.dim() == 3:
            mask_pred = mask_pred[None]

        # mask_pred = mask_pred.detach().numpy().copy()

        foreground_pred = torch.sigmoid(mask_pred[..., 0])
        angles_pred = torch.sigmoid(mask_pred[..., 1]) * 2 * np.pi - np.pi
        # class_pred = softmax(mask_pred[...,2:], axis=-1)

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

            # assert im_true.min() >= 0 and im_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # # convert to one-hot format
            # im_true = F.one_hot(im_true, net.n_classes).permute(0, 3, 1, 2).float()
            # im_pred = F.one_hot(im_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # # compute the Dice score, ignoring background
            # dice_score += multiclass_dice_coeff(im_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=True)

    net.train()
    return dice_score / max(num_val_batches, 1)


def plot_predictions(net, dataloader, idx=0, threshold=0.5):
    for i, batch in enumerate(dataloader):
        if i < idx:
            continue
        x, y, z, labels, angles, classes, label_mask, nucleus_mask = (
            batch["X"],
            batch["Y"],
            batch["gene"],
            batch["labels"].numpy().copy().astype(int),
            batch["angles"].numpy().copy().astype(float),
            batch["classes"].numpy().copy().astype(int),
            batch["label_mask"].numpy().copy().astype(bool),
            batch["nucleus_mask"].numpy().copy().astype(bool),
        )
        break

    net.eval()
    mask_pred = net(x, y, z).detach().numpy().copy()
    foreground_pred = expit(mask_pred[..., 0])
    angles_pred = expit(mask_pred[..., 1]) * 2 * np.pi - np.pi
    class_pred = softmax(mask_pred[..., 2:], axis=-1)

    x, y, z = x.numpy().copy(), y.numpy().copy(), z.numpy().copy()
    angles = angles * 2 * np.pi - np.pi

    # Setup the colored nuclei
    label_plot = np.array(labels.astype(float))
    for i, c in enumerate(
        np.random.choice(
            np.unique(label_plot[label_mask]),
            replace=False,
            size=int(label_plot[label_mask].max() + 1),
        )
    ):
        label_plot[label_plot == c] = (i % 16) + 4
    label_plot[labels == 0] = 0
    # label_plot[label_mask] = (label_plot[label_mask] > 0).astype(int)
    label_plot[~label_mask] = np.nan

    # Determine the label predictions
    pred_plot = np.zeros_like(foreground_pred)
    pred_mask = foreground_pred >= threshold
    pred_plot[pred_mask] = class_pred.argmax(axis=-1)[pred_mask] + 1
    for i, c in enumerate(np.unique(pred_plot[pred_mask])):
        pred_plot[pred_plot == c] = (i % 16) + 4

    fig, axarr = plt.subplots(
        3, x.shape[0], figsize=(5 * x.shape[0], 10), sharex=True, sharey=True
    )
    for i in range(x.shape[0]):
        axarr[0, i].set_title("Labels and transcripts")
        axarr[0, i].imshow(label_plot[i], cmap="tab20b", interpolation="none")
        axarr[0, i].scatter(y[i], x[i], color="gray", zorder=100, s=1)

        axarr[1, i].set_title("Predicted labels")
        axarr[1, i].imshow(pred_plot[i], cmap="tab20b", interpolation="none")

        axarr[2, i].imshow(
            foreground_pred[i], vmin=0, vmax=1, cmap="coolwarm", interpolation="none"
        )
        axarr[2, i].set_title("Predicted probabilities")

        for xi in range(label_plot.shape[1]):
            for yi in range(label_plot.shape[2]):
                if nucleus_mask[i, xi, yi]:
                    dx, dy = pol2cart(0.5, angles[i, xi, yi])
                    axarr[0, i].arrow(yi + 0.5, xi + 0.5, dy, dx, width=0.07, alpha=0.5)

                if pred_mask[i, xi, yi]:
                    dx, dy = pol2cart(0.5, angles_pred[i, xi, yi])
                    axarr[1, i].arrow(yi + 0.5, xi + 0.5, dy, dx, width=0.07, alpha=0.5)

    plt.show()


def score_segmentation(segments, nuclei):
    # Get the nuclei pixels and the label assigned to each pixel
    nuclei_mask = nuclei >= 0
    nuclei_uniques, inv_map, nuclei_counts = np.unique(
        nuclei[nuclei_mask], return_inverse=True, return_counts=True
    )
    n_nuclei = len(nuclei_uniques)
    nuclei_segments = segments[nuclei_mask]

    # Assess the fraction of each nucleus labeled as foreground
    percent_foreground = np.zeros(n_nuclei)
    np.add.at(percent_foreground, inv_map[nuclei_segments >= 0], 1)
    percent_foreground = percent_foreground / nuclei_counts

    # Calculate some per-nuclei statistics
    percent_common = np.zeros(n_nuclei)
    nuclei_label_counts = np.zeros(n_nuclei)
    label_nuclei_counts = np.zeros(np.unique(nuclei_segments)[-1] + 1)
    for i in tqdm.trange(n_nuclei, desc="Scoring nuclei"):

        local_labels = nuclei_segments[inv_map == i]
        local_uniques, local_counts = np.unique(
            local_labels[local_labels >= 0], return_counts=True
        )

        # Edge case: no labeled pixels. everything is zero.
        if len(local_uniques) == 0:
            continue

        # Assess the fraction of each nucleus labeled with the dominant cell ID
        percent_common[i] = np.max(local_counts) / len(local_labels)

        # Assess the fraction of nuclei labeled with more than a single cell ID
        nuclei_label_counts[i] = len(local_uniques) > 1

        # Assess the fraction of each label appearing in more than a single nucleus
        label_nuclei_counts[local_uniques] = label_nuclei_counts[local_uniques] + 1

    return {
        "percent_foreground": percent_foreground,
        "percent_common": percent_common,
        "nuclei_label_counts": nuclei_label_counts,
        "label_nuclei_counts": label_nuclei_counts,
    }

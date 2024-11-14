import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from scipy.special import softmax, expit
from torch import Tensor

from nuc2seg.data import collate_tiles
from nuc2seg.preprocessing import pol2cart


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

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


def plot_predictions(net, dataloader, idx=0, threshold=0.5):
    batch = collate_tiles([dataloader[idx]])
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
        nrows=3,
        ncols=x.shape[0],
        figsize=(5 * x.shape[0], 10),
        sharex=True,
        sharey=True,
    )
    if len(axarr.shape) == 1:
        axarr = axarr[:, None]

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


def foreground_accuracy(prediction, labels):
    mask = labels > 0
    target = (labels > 0).float()

    foreground_pred = torch.sigmoid(prediction[..., 0])
    foreground_pred = torch.where(mask, foreground_pred, 0)

    return dice_coeff(foreground_pred, target)


def squared_angle_difference(predictions, targets):
    """Angles are expressed in [0,1] but we want 0.01 and 0.99 to be close.
    So we take the minimum of the losses between the original prediction,
    adding 1, and subtracting 1 such that we consider 0.01, 1.01, and -1.01.
    That way 0.01 and 0.99 are only 0.02 apart."""
    delta = predictions - targets
    return torch.minimum(torch.minimum(delta**2, (delta - 1) ** 2), (delta + 1) ** 2)


def angle_accuracy(predictions, labels, target):
    mask = labels > 0
    angle_pred = torch.sigmoid(predictions[..., 1])

    angle_pred = angle_pred[mask]
    target = target[mask]

    return (
        torch.tensor(1.0)
        - torch.sqrt(squared_angle_difference(angle_pred, target)).mean()
    )


def celltype_accuracy(predictions, labels):
    mask = labels > 0
    class_pred = torch.softmax(predictions[..., 2:], dim=-1).argmax(dim=-1)
    n_classes = predictions.shape[-1] - 2

    labels = labels[mask] - 1
    class_pred = class_pred[mask]

    mask_true = F.one_hot(labels, n_classes)
    mask_pred = F.one_hot(class_pred, n_classes)
    return multiclass_dice_coeff(mask_pred[..., None], mask_true[..., None])

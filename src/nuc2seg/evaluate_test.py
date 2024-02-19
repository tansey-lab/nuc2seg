from nuc2seg.evaluate import (
    dice_coeff,
    multiclass_dice_coeff,
    foreground_accuracy,
    squared_angle_difference,
    angle_accuracy,
)
import torch


def test_dice_coeff():
    input = torch.tensor([[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]])
    result = dice_coeff(input, target)
    assert result.item() < 1.0

    input = torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]])
    result = dice_coeff(input, target)
    assert result.item() == 1.0

    input = torch.tensor([[[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]])
    target = torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]])
    result = dice_coeff(input, target)
    assert result.item() < 1.0


def test_multiclass_dice_coeff():
    input = torch.tensor(
        [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]]]
    )
    target = torch.tensor(
        [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]]]
    )
    result = multiclass_dice_coeff(input, target)
    assert result.item() == 1.0

    input = torch.tensor(
        [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]]
    )
    target = torch.tensor(
        [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]]]
    )
    result = multiclass_dice_coeff(input, target)
    assert result.item() < 1.0


def test_foreground_accuracy():
    prediction = torch.tensor([[[[0.5, 0.5, 0.1]]]])
    prediction = prediction.permute(0, 2, 3, 1)
    labels = torch.tensor([[[1, 1, 0]]])
    result = foreground_accuracy(prediction, labels)

    assert 0 < result.item() < 1

    prediction = torch.tensor([[[[100, 100, -100]]]])
    prediction = prediction.permute(0, 2, 3, 1)
    labels = torch.tensor([[[1, 1, 0]]])
    result = foreground_accuracy(prediction, labels)

    assert torch.isclose(result, torch.tensor(1.0))

    prediction = torch.tensor([[[[100, 100, 100]]]])
    prediction = prediction.permute(0, 2, 3, 1)
    labels = torch.tensor([[[1, 1, 0]]])
    result = foreground_accuracy(prediction, labels)

    assert torch.isclose(result, torch.tensor(1.0))


def test_foreground_accuracy_multi_batch():
    prediction = torch.tensor([[[[100, 100, -100]]], [[[100, 100, -100]]]])
    prediction = prediction.permute(0, 2, 3, 1)
    labels = torch.tensor([[[1, 1, 0]], [[1, 1, 0]]])
    result = foreground_accuracy(prediction, labels)

    assert torch.isclose(result, torch.tensor(1.0))

    prediction = torch.tensor([[[[100, 100, 100]]], [[[100, 100, 100]]]])
    prediction = prediction.permute(0, 2, 3, 1)
    labels = torch.tensor([[[1, 1, 0]], [[1, 1, 0]]])
    result = foreground_accuracy(prediction, labels)

    assert torch.isclose(result, torch.tensor(1.0))


def test_angle_difference():
    result = squared_angle_difference(torch.tensor([0.0]), torch.tensor([0.0]))

    assert result.item() == 0.0

    result = squared_angle_difference(torch.tensor([0.0]), torch.tensor([0.01]))

    assert torch.allclose(result, torch.tensor([0.01**2]))

    result = squared_angle_difference(torch.tensor([0.99]), torch.tensor([0.01]))

    assert torch.allclose(result, torch.tensor([0.02**2]))


def test_angle_accuracy():
    prediction = torch.zeros(1, 2, 2, 3)  # batch  # X  # Y  # channels

    prediction[0, 0, 0, 1] = torch.logit(torch.tensor(0.99))
    prediction[0, 0, 1, 1] = torch.logit(torch.tensor(0.99))
    prediction[0, 1, 0, 1] = torch.logit(torch.tensor(0.01))
    prediction[0, 1, 1, 1] = torch.logit(torch.tensor(0.5))

    labels = torch.tensor([[[-1, -1], [-1, 1]]])
    target = torch.tensor([[[0.01, 0.01], [0.99, 0.99]]])

    result = angle_accuracy(prediction, labels, target)

    assert torch.allclose(result, torch.tensor(0.98))

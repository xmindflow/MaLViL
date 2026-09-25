
import torch
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
from utils import calculate_metric_percase, calculate_dice_percase
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler


def select_mid_slice_indices(n_slices: int, k: int, seed: int = 1234) -> np.ndarray:
    """Evenly sample `k` slices from a wide mid-volume band (fallback)."""
    if n_slices <= 0:
        return np.array([], dtype=np.int64)
    k = int(max(1, min(k, n_slices)))
    if k >= n_slices:
        return np.arange(n_slices, dtype=np.int64)

    # Use central 90% (skip ~5% ends) — empty edge slices hurt rare-organ coverage.
    lo = int(round(0.05 * (n_slices - 1)))
    hi = int(round(0.95 * (n_slices - 1)))
    if hi <= lo:
        lo, hi = 0, n_slices - 1

    band = np.arange(lo, hi + 1, dtype=np.int64)
    if len(band) <= k:
        return band

    pos = np.linspace(0, len(band) - 1, num=k)
    idx = np.unique(np.round(pos).astype(np.int64))
    if len(idx) < k:
        rng = np.random.RandomState(seed)
        remain = np.setdiff1d(np.arange(len(band)), idx, assume_unique=False)
        need = k - len(idx)
        if len(remain) > 0:
            extra = rng.choice(remain, size=min(need, len(remain)), replace=False)
            idx = np.sort(np.concatenate([idx, extra]))
    return band[idx]


def select_fast_slice_indices(
    label,
    k: int,
    num_classes: int,
    seed: int = 1234,
) -> np.ndarray:
    """Pick `k` slices for fast validation, covering every organ in the volume.

    Ground truth is used only to choose which slices to score, so thin organs
    are not missed. It is not used to change the prediction. Reported Synapse
    numbers are a full-volume pass over the same 12 validation cases.
    """
    lab = np.asarray(label)
    if lab.ndim == 4:  # [1, D, H, W]
        lab = lab[0]
    if lab.ndim != 3:
        raise ValueError(f"Expected label [D,H,W], got shape {lab.shape}")

    D = lab.shape[0]
    k = int(max(1, min(k, D)))
    if k >= D:
        return np.arange(D, dtype=np.int64)

    class_hits = {}
    for c in range(1, int(num_classes)):
        hits = np.where(np.any(lab == c, axis=(1, 2)))[0]
        if len(hits) > 0:
            class_hits[c] = hits.astype(np.int64)

    if not class_hits:
        return select_mid_slice_indices(D, k, seed=seed)

    selected = []

    def _take_even(hits: np.ndarray, n: int):
        n = int(max(1, min(n, len(hits))))
        if n == 1:
            return [int(hits[len(hits) // 2])]
        pos = np.linspace(0, len(hits) - 1, num=n)
        return [int(hits[int(round(p))]) for p in pos]

    # Pass 1: guarantee ≥1 slice per present organ (middle of its Z-extent).
    for c in sorted(class_hits):
        selected.extend(_take_even(class_hits[c], 1))
    selected = list(dict.fromkeys(selected))

    # Pass 2: give rare / thin organs extra slices first, then larger ones.
    rare_first = sorted(class_hits, key=lambda c: len(class_hits[c]))
    per_class_budget = max(1, (k + len(rare_first) - 1) // max(len(rare_first), 1))
    for c in rare_first:
        if len(selected) >= k:
            break
        for idx in _take_even(class_hits[c], per_class_budget):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break

    # Pass 3: fill remaining with even global coverage.
    if len(selected) < k:
        for idx in select_mid_slice_indices(D, k, seed=seed):
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= k:
                break

    return np.array(sorted(selected[:k]), dtype=np.int64)


def _metric_or_nan(pred_mask, gt_mask):
    """Like calculate_metric_percase, but NaN when the class is absent from GT subset."""
    gt = gt_mask.astype(bool)
    if not gt.any():
        return (np.nan, np.nan, np.nan, np.nan)
    return calculate_metric_percase(pred_mask.copy(), gt_mask.copy())


def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1,
    epoch=0,
    slice_indices=None,
):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    full_volume = slice_indices is None
    if full_volume:
        slice_indices = range(image.shape[0])
    else:
        slice_indices = [int(i) for i in slice_indices]

    last_pred = None
    for ind in slice_indices:
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = net(input)
            if isinstance(outputs, tuple): # reconstruction output for calculating reconstruction loss
                outputs = outputs[0]
            if isinstance(outputs, list): # for deep supervision outputs
                outputs = outputs[-1]
            out = torch.argmax(outputs, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
            last_pred = pred

    # Fast-test: score only sampled slices. Absent classes → NaN (excluded from mean).
    if not full_volume and len(slice_indices) < image.shape[0]:
        pred_sub = prediction[slice_indices]
        label_sub = label[slice_indices]
        metric_list = []
        for i in range(1, classes):
            metric_list.append(_metric_or_nan(pred_sub == i, label_sub == i))
    else:
        metric_list = []
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

    return last_pred if last_pred is not None else prediction, metric_list


def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = net(input)
                if isinstance(outputs, list): # for deep supervision outputs
                    outputs = outputs[-1]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = net(input)
            if isinstance(outputs, tuple): # reconstruction output for calculating reconstruction loss
                outputs = outputs[0]
            if isinstance(outputs, list): # for deep supervision outputs
                outputs = outputs[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction==i, label==i))
    return metric_list

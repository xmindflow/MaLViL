import copy
import io
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stderr
import torch
import torch.nn.functional as F
from medpy import metric


IMGNET_MEAN=[0.485, 0.456, 0.406]
IMGNET_STD=[0.229, 0.224, 0.225]

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.dc(pred, gt)
        hd95 = metric.hd95(pred, gt)
        jaccard = metric.jc(pred, gt)
        asd = metric.asd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0


@torch.no_grad()
def compute_segmentation_metrics_hard(
    logits: torch.Tensor,              # [B,C,H,W]
    labels: torch.Tensor,              # [B,H,W] or [B,1,H,W]
    ignore_index: int | None = None,   # e.g. 255
    include_background: bool = False,  # typically False for "mean"
    eps: float = 1e-7,
):
    """
    Returns TRUE metric values (not errors):
      - per_class_dice, mean_dice
      - per_class_iou, mean_iou
      - per_class_recall, mean_recall
      - pixel_accuracy
    All computed as HARD metrics on argmax prediction.
    Averages are per-image, then mean across batch (and can be averaged across dataset by caller).
    """

    if isinstance(logits, tuple): # for (outputs, rec_error) in some models
        logits = logits[0]
    if isinstance(logits, list): # for deep supervision outputs
        logits = logits[-1]

    if labels.ndim == 4 and labels.size(1) == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"labels must be [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}")

    B, C, H, W = logits.shape
    if labels.shape != (B, H, W):
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")

    labels = labels.long()

    # valid mask
    if ignore_index is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid = (labels != ignore_index)

    # prediction (hard)
    pred = logits.argmax(dim=1)  # [B,H,W]

    # clamp invalid labelss only for comparisons (do NOT change valid ones)
    # (we won't one-hot; we'll use boolean masks)

    classes = list(range(C))
    if not include_background and C > 1:
        classes = classes[1:]  # drop background=0 from "mean"

    per_class_dice = {}
    per_class_iou = {}
    per_class_accuracy = {}
    per_class_hd95 = {}
    per_class_jaccard = {}
    per_class_asd = {}

    # compute per-image metrics then average (standard, stable)
    for cls in range(C):
        dice_b = []
        iou_b = []
        acc_b = []
        hd95_b = []
        jaccard_b = []
        asd_b = []

        for b in range(B):
            vb = valid[b]
            pb = (pred[b] == cls) & vb
            gb = (labels[b] == cls) & vb

            tp = (pb & gb).sum().float()
            fp = (pb & ~gb).sum().float()
            fn = (~pb & gb).sum().float()
            tn = ((~pb) & (~gb) & vb).sum().float()

            # For per-case metrics, we can compute on the CPU and use medpy for stable Dice, HD95, Jaccard, ASD, etc.
            dice, hd95, jaccard, asd = calculate_metric_percase(pb.cpu().numpy(), gb.cpu().numpy())

            # IoU = TP / (TP + FP + FN)
            denom_i = (tp + fp + fn)
            iou = (tp + eps) / (denom_i + eps)
            # Pixel accuracy = (TP + TN) / (TP + TN + FP + FN)  (was previously IoU by mistake)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
            
            dice_b.append(dice)
            iou_b.append(iou.item())
            acc_b.append(acc.item())
            hd95_b.append(hd95)
            jaccard_b.append(jaccard)
            asd_b.append(asd)

        per_class_accuracy[cls] = float(np.array(acc_b).mean())
        per_class_hd95[cls] = float(np.array(hd95_b).mean())
        per_class_jaccard[cls] = float(np.array(jaccard_b).mean())
        per_class_asd[cls] = float(np.array(asd_b).mean())
        per_class_dice[cls] = float(np.array(dice_b).mean())
        per_class_iou[cls] = float(np.array(iou_b).mean())

    # mean over selected classes (ignore NaNs)
    def mean_over_classes(vals):
        vals = [v for c, v in vals.items() if c in classes]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) == 0:
            return float('nan')
        return float(np.mean(vals))

    mean_dice = mean_over_classes(per_class_dice)
    mean_iou = mean_over_classes(per_class_iou)
    mean_accuracy = mean_over_classes(per_class_accuracy)
    mean_hd95 = mean_over_classes(per_class_hd95)
    mean_jaccard = mean_over_classes(per_class_jaccard)
    mean_asd = mean_over_classes(per_class_asd)

    return {
        "per_class_dice": per_class_dice,
        "mean_dice": mean_dice,
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "per_class_accuracy": per_class_accuracy,
        "mean_accuracy": mean_accuracy,
        "per_class_hd95": per_class_hd95,
        "mean_hd95": mean_hd95,
        "per_class_jaccard": per_class_jaccard,
        "mean_jaccard": mean_jaccard,
        "per_class_asd": per_class_asd,
        "mean_asd": mean_asd,
        "num_classes": C,
        "ignore_index": ignore_index,
        "include_background_in_mean": include_background,
    }



def compute_metrics(pred, target, threshold=0.5, eps=1e-7):
    """
    Compute Dice, mIoU, ACC, and MAE for binary segmentation.

    Args:
        pred:  Tensor of shape (B, 1, H, W)
               raw logits OR probabilities OR binary predictions
        target: Tensor of shape (B, 1, H, W)
                binary ground truth {0,1}
        threshold: threshold for binarizing probabilities
        eps: numerical stability constant

    Returns:
        dice: float
        miou: float
        acc: float
        mae: float
    """
    if isinstance(pred, list): pred = pred[-1]  # take last output if multiple outputs are given (as in deep supervision)

    # --- 1) Convert logits -> probabilities if needed ---
    if pred.dtype.is_floating_point:
        if pred.max() > 1.0 or pred.min() < 0.0:
            # assume raw logits → convert to sigmoid proba
            pred = torch.sigmoid(pred)

    # --- 2) Binarize prediction ---
    pred_bin = (pred >= threshold).float()
    target_bin = (target > 0.5).float()

    # Flatten for global computation
    pred_f = pred_bin.view(-1)
    target_f = target_bin.view(-1)

    # --- 3) Dice ---
    intersection = (pred_f * target_f).sum()
    dice = (2 * intersection + eps) / (pred_f.sum() + target_f.sum() + eps)

    # --- 4) IoU ---
    union = pred_f.sum() + target_f.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    miou = iou  # binary segmentation → single IoU = mIoU

    # --- 5) Pixel Accuracy ---
    correct = (pred_f == target_f).sum()
    total = pred_f.numel()
    acc = correct.float() / total

    # --- 6) MAE (non-binarized prediction vs binary target) ---
    mae = F.l1_loss(pred.view(-1), target_bin.view(-1))

    # Return python floats
    return dice.item(), miou.item(), acc.item(), mae.item()


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()

    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)

    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]

    return input_flatten, target_flatten

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image

def vertical_flip(image):
    image = image[::-1, :, :]
    return image

def tta_model(model, image):
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_mask = model.predict(np.expand_dims(n_image, axis=0))[0]
    h_mask = model.predict(np.expand_dims(h_image, axis=0))[0]
    v_mask = model.predict(np.expand_dims(v_image, axis=0))[0]

    n_mask = n_mask
    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)

    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    return mean_mask

def CalParams(model, input_tensor, logger=None):
    # Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    from thop import profile
    from thop import clever_format
    try:
        model_copy = copy.deepcopy(model)  # Prevent thop from modifying the original
    except:
        model_copy = copy.copy(model) 
    model_copy.eval()

    flops, params = profile(model_copy, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'[Statistics Information] FLOPs: {flops}, Params: {params}')
    if logger is not None:
        logger.info(f'[Statistics Information] FLOPs: {flops}, Params: {params}')

def print_param_flops(net, args):
    from fvcore.nn import FlopCountAnalysis
    net.eval()  # Ensure model is in eval mode
    dummy_input = torch.randn(1, args.input_channels, args.img_size, args.img_size).cuda()

    with torch.no_grad():
        with redirect_stderr(io.StringIO()):
            f = FlopCountAnalysis(net, dummy_input)
            print(f'Model parameters: {sum([m.numel() for m in net.parameters()])}, FLOPs: {f.total()/1e9:.2f}G')
            if hasattr(net, 'backbone'):
                print(f' - Backbone <pvt_v2_b2> params: {sum([m.numel() for m in net.backbone.parameters()])}')
            if hasattr(net, 'encoder'):
                print(f' - Encoder params: {sum([m.numel() for m in net.encoder.parameters()])}')
            if hasattr(net, 'decoder'):
                print(f' - Decoder params: {sum([m.numel() for m in net.decoder.parameters()])}')
            print(f' --> Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.3f}M')

    # Optional: warm-up model to restore CUDA performance
    for _ in range(5):
        _ = net(dummy_input)

def print_model_stats(model, input_size=(3, 224, 224)):
    from ptflops import get_model_complexity_info
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model created, param count: {total_params}')
    
    # Calculate GMACs using ptflops
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    
    # Display GMACs and params
    print(f'Model: {macs} GMACs, {params} parameters')
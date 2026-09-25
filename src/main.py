import os, sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils import Criterion, get_optimizer, get_scheduler, print_param_flops
from train_test import train
from utils.utils_skin import test, plot_result, binary_save_test_images


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="MaLViL (paper release)")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="ISIC2017, ISIC2018, PH2, HAM10000, BUSI, SYNAPSE")
parser.add_argument("--data_dir", default="/path/to/datasets")
parser.add_argument("--save_path", default="./results")
parser.add_argument("--encoder_ptdir", default=os.path.join(_REPO, "pretrained_pth"),
                    help="directory containing pvt/pvt_v2_b2.pth")
parser.add_argument("--checkpoint", default=None, help="weights for --eval or warm-start")
parser.add_argument("--resume", type=str, nargs="?", const="", default=None,
                    help="resume from last.pth (--resume) or an explicit path")
parser.add_argument("--tag", default="malvil", help="experiment tag")
parser.add_argument("--model_name", type=str, default="malvil")
parser.add_argument("--eval", action="store_true", help="evaluation only")
parser.add_argument("--save_test", action="store_true", help="write prediction visualizations")

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=40)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--input_channels", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--amp", action="store_true", help="mixed precision")
parser.add_argument("--z_spacing", type=int, default=10, help="Synapse HD95 spacing")
parser.add_argument("--test_save_dir", default="./predictions")
parser.add_argument("--val_interval", type=int, default=1)

parser.add_argument("--optimizer", type=str, default="AdamW", help="AdamW | Adam | SGD")
parser.add_argument("--scheduler", type=str, default="poly", help="poly | cosine")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_enc", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--loss_type", type=str, default="dice,ce")
parser.add_argument("--loss_weights", type=str, default="0.5,0.5")
parser.add_argument("--lambda_rec_loss", type=float, default=0.01,
                    help="Bi-LRViL reconstruction loss weight (paper: 0.01)")
parser.add_argument("--max_grad_norm", type=float, default=0.5,
                    help="clip_grad_norm max_norm; 0 disables clipping")


args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.tag = args.tag or "malvil"

snapshot_path = f"{args.save_path}/{args.tag}"
snapshot_path = snapshot_path + '_ep' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_lrenc' + str(args.lr_enc) if args.lr_enc != 1e-5 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.tag)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
writer = SummaryWriter(snapshot_path + '/log')

_log_fn = "eval"if args.eval else "train"
logging.basicConfig(
    filename=snapshot_path + "/log_" + _log_fn + ".txt", 
    level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
if args.eval:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))
log_filename = f'{snapshot_path}' + '/log_' + _log_fn + '.txt'

from pprint import pprint
pprint(vars(args))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# -------------- model ----------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("Building model...")
from utils.builder import build_model
net = build_model(args, writer=writer, device=device, logger=logging)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# -------------- model ----------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


print_param_flops(net, args)


# ----------------- dataset --------------------
# preparing training dataset
from utils.dataloader import get_dataloaders
print("Preparing datasets...")
tr_loader, vl_loader, te_loader = get_dataloaders(args, logging)
# ---------------------------------------------


def _load_checkpoint_weights(net, path):
    """Load a raw state dict or a training checkpoint (`model` / `model_state_dict`)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    net.load_state_dict(ckpt)


if args.eval:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation mode")
    net.eval()
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file {args.checkpoint} does not exist.")
        _load_checkpoint_weights(net, args.checkpoint)
    elif os.path.exists(os.path.join(snapshot_path, "best.pth")):
        print(f"Loading best weights from {os.path.join(snapshot_path, 'best.pth')}")
        _load_checkpoint_weights(net, os.path.join(snapshot_path, "best.pth"))
    else:
        print("No weights file provided...")
        exit(0)
    test_save_dir = os.path.join(snapshot_path, "predictions")
    os.makedirs(test_save_dir, exist_ok=True)

    print(f"Testing on datasets...")
    if args.dataset_name.lower() == "synapse":
        from train_test import inference_3d
        save_pred = test_save_dir if args.save_test else None
        _, te_avg_dcs, te_hd95 = inference_3d(
            args, net, te_loader, save_pred, epoch=args.max_epochs, logging=logging, fast_val=False,
        )
        print(
            f"\t {args.dataset_name} dataset -> Val-3D <{args.tag}> -> "
            f"mean Dice: {te_avg_dcs:.4f}, mean HD95: {te_hd95:.4f}"
        )
    elif isinstance(te_loader, list):
        for te_name, te_l in te_loader:
            te_avg_dcs, te_avg_acc, te_avg_iou = test(net, te_l, logging, -1, save_path=os.path.join(test_save_dir, te_name))
            print(f"\t {args.dataset_name} dataset [{te_name}] -> Test <{args.tag}> -> Average Dice: {te_avg_dcs:.4f}, Average Accuracy: {te_avg_acc:.4f}, Average IoU: {te_avg_iou:.4f}")
        if args.save_test:
            print("Saving test images...")
            for te_name, te_l in te_loader:
                binary_save_test_images(net, te_l, os.path.join(snapshot_path, "predictions", te_name), device="cuda")
    else:
        te_avg_dcs, te_avg_acc, te_avg_iou = test(net, te_loader, logging, -1, save_path=args.test_save_dir)
        print(f"\t {args.dataset_name} dataset -> Test <{args.tag}> -> Average Dice: {te_avg_dcs:.4f}, Average Accuracy: {te_avg_acc:.4f}, Average IoU: {te_avg_iou:.4f}")
        if args.save_test:
            print("Saving test images...")
            binary_save_test_images(net, te_loader, os.path.join(snapshot_path, "predictions"), device=device)
    exit(0)


# Warm-start weights only (not a full training resume). Ignored when --resume is set.
if args.checkpoint and args.resume is None:
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint} does not exist.")
    net.load_state_dict(torch.load(args.checkpoint, weights_only=True))


if args.n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()


# training mode
print("Training mode")
net.train()
best_net = net


criterion = Criterion(args.num_classes, args)
if args.amp:
    try:
        from torch.amp import autocast, GradScaler
    except ImportError:
        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler
    scaler = GradScaler()
    print("AMP enabled...")
else:
    print("AMP disabled...")
    scaler = None

iter_num = 0

Loss = []
te_accuracy = []

best_dcs_vl = 0
best_dcs_te = 0
dice_ = []

max_iterations = args.max_epochs * len(tr_loader)

optimizer = get_optimizer(net, args)
scheduler = get_scheduler(optimizer, args, max_iterations=args.max_epochs*len(tr_loader))

# Full training resume (crash / new machine): restore model + optimizer + scheduler + epoch.
resume_state = None
if args.resume is not None:
    resume_path = (
        os.path.join(snapshot_path, "last.pth")
        if args.resume in ("", "auto")
        else args.resume
    )
    if not os.path.exists(resume_path):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resume_path}. "
            "Copy the run folder (or last.pth) and use the same --tag / hyperparams, "
            "or pass --resume /path/to/last.pth"
        )
    print(f"Resuming training from {resume_path}")
    logging.info(f"Resuming training from {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"{resume_path} is not a full training checkpoint. "
            "Expected keys like model/optimizer/scheduler/epoch (saved as last.pth). "
            "For weights-only init use --checkpoint instead."
        )
    model_ref = net.module if hasattr(net, "module") else net
    model_ref.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    resume_state = {
        "start_epoch": int(ckpt.get("epoch", 0)),
        "global_iter_num": int(ckpt.get("global_iter_num", 0)),
        "best_dcs_vl": float(ckpt.get("best_dcs_vl", 0.0)),
        "best_dcs_te": float(ckpt.get("best_dcs_te", 0.0)),
        "best_ep_idx": int(ckpt.get("best_ep_idx", -1)),
        "best_dcs_fast": float(ckpt.get("best_dcs_fast", 0.0)),
        "best_ep_fast": int(ckpt.get("best_ep_fast", -1)),
    }
    _is_synapse = args.dataset_name.lower() == "synapse"
    _best_name = "best val Dice"
    _resume_msg = (
        f"Resumed at epoch {resume_state['start_epoch']}/{args.max_epochs} "
        f"({_best_name}={resume_state['best_dcs_te']:.4f}"
        + (f", best fast Dice={resume_state['best_dcs_fast']:.4f}" if _is_synapse else "")
        + ")"
    )
    print(_resume_msg)
    logging.info(_resume_msg)

print("Starting training...")

loaders = {'train': tr_loader, 'val': vl_loader, 'test': te_loader}
train(
    args, net, loaders, optimizer, scheduler, criterion, logging, snapshot_path, writer, scaler,
    resume_state=resume_state,
)

# Loading the best model (weights-only best.pth; may be missing if never tested)
best_path = os.path.join(snapshot_path, 'best.pth')
if os.path.exists(best_path):
    best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
    model_ref = best_net.module if hasattr(best_net, "module") else best_net
    try:
        model_ref.load_state_dict(best_checkpoint)
    except RuntimeError:
        # Older runs may have saved DataParallel keys ("module.*").
        best_net.load_state_dict(best_checkpoint)
else:
    print(f"Warning: no best.pth at {best_path}; keeping last training weights.")
    logging.info(f"Warning: no best.pth at {best_path}; keeping last training weights.")

if args.save_test:
    print("Saving test images...")
    if isinstance(te_loader, list):
        for te_name, te_l in te_loader:
            binary_save_test_images(best_net, te_l, os.path.join(snapshot_path, "visualized", te_name), device=device)
    else:
        if args.dataset_name.lower() == "synapse":
            from train_test import inference_3d
            test_save_path = os.path.join(snapshot_path, "visualized")
            os.makedirs(test_save_path, exist_ok=True)
            inference_3d(args, best_net, te_loader, test_save_path, epoch=args.max_epochs, logging=logging, fast_val=False)
        else:
            binary_save_test_images(best_net, te_loader, os.path.join(snapshot_path, "visualized"), device=device)

plot_result(dice_, dice_, snapshot_path, args)
writer.close()

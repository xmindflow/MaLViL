#!/usr/bin/env bash
# MaLViL launcher.
#   bash scripts/main.sh DATASET TRAIN [--tag name] [--resume]
#   EVAL_PT=/path/to/ckpt.pth bash scripts/main.sh DATASET TEST
#
# Datasets: ISIC2017 | ISIC2018 | PH2 | HAM10000 | BUSI | SYNAPSE
#
# Shared defaults (override with the flags below):
#   AdamW, poly, AMP, batch size 8, lr = lr_enc = 1e-4, seed 1234
#   image 224, loss dice+ce (0.5, 0.5), grad clip 0.5
# Per dataset:
#   ISIC2017, ISIC2018, HAM10000   40 epochs, 224, dice+ce
#   PH2                            150 epochs, 224, dice+ce
#   BUSI                           50 epochs, 256, 1 channel, dice+ce
#   SYNAPSE                        350 epochs, 224, 1 channel, 9 classes, boundary loss
set -euo pipefail

model_name="malvil"
max_epochs=""
batch_size=8
lr=0.0001
lr_enc=0.0001
optimizer="AdamW"
scheduler="poly"
loss_type="dice,ce"
loss_weights="0.5,0.5"
num_classes=2
input_channels=3
IMG_SIZE=224
resume=""
tag=""
seed=1234
max_grad_norm=0.5

DATASET=${1:?dataset required}
MODE=${2:?mode TRAIN|TEST required}
shift 2

while [[ $# -gt 0 ]]; do
  case $1 in
    --lr) lr="$2"; shift 2 ;;
    --lr_enc) lr_enc="$2"; shift 2 ;;
    --optimizer) optimizer="$2"; shift 2 ;;
    --scheduler) scheduler="$2"; shift 2 ;;
    --loss_type) loss_type="$2"; shift 2 ;;
    --loss_weights) loss_weights="$2"; shift 2 ;;
    --max_epochs) max_epochs="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --img_size) IMG_SIZE="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --max_grad_norm) max_grad_norm="$2"; shift 2 ;;
    --resume)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then resume="$2"; shift 2
      else resume="auto"; shift 1; fi ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Paths: set DATA_DIR / RESULTS_DIR in the environment, or edit the defaults.
PROG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/path/to/datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROG_DIR}/results}"
PT_DIR="${PROG_DIR}/pretrained_pth"
EVAL_PT="${EVAL_PT:-}"

case "$DATASET" in
  HAM10000)
    data_dir="${DATA_DIR}/Skin/HAM10000"
    save_path="${RESULTS_DIR}/ham"
    max_epochs="${max_epochs:-40}"
    ;;
  PH2)
    data_dir="${DATA_DIR}/Skin/PH2"
    save_path="${RESULTS_DIR}/ph2"
    max_epochs="${max_epochs:-150}"
    ;;
  ISIC2017)
    data_dir="${DATA_DIR}/Skin/ISIC2017"
    save_path="${RESULTS_DIR}/isic2017"
    max_epochs="${max_epochs:-40}"
    ;;
  ISIC2018)
    data_dir="${DATA_DIR}/Skin/ISIC2018"
    save_path="${RESULTS_DIR}/isic2018"
    max_epochs="${max_epochs:-40}"
    ;;
  BUSI)
    data_dir="${DATA_DIR}/US/busi"
    save_path="${RESULTS_DIR}/busi"
    input_channels=1
    num_classes=2
    max_epochs="${max_epochs:-50}"
    IMG_SIZE=256
    ;;
  SYNAPSE)
    data_dir="${DATA_DIR}/Synapse"
    save_path="${RESULTS_DIR}/synapse"
    input_channels=1
    num_classes=9
    loss_type=boundary
    loss_weights=1.0
    max_epochs="${max_epochs:-350}"
    ;;
  *)
    echo "Invalid dataset: $DATASET"
    echo "Valid: HAM10000 PH2 ISIC2017 ISIC2018 BUSI SYNAPSE"
    exit 1
    ;;
esac

tag="Full-${model_name}_${DATASET}${tag:+_$tag}"
cd "${PROG_DIR}/src"

echo "==== CONFIG ===="
echo "DATASET=$DATASET MODE=$MODE epochs=$max_epochs bs=$batch_size img=$IMG_SIZE"
echo "optimizer=$optimizer scheduler=$scheduler lr=$lr lr_enc=$lr_enc seed=$seed"
echo "loss=$loss_type weights=$loss_weights channels=$input_channels classes=$num_classes"
echo "==============="

common=(
  --dataset_name "${DATASET}"
  --model_name "${model_name}"
  --tag "${tag}"
  --max_epochs "${max_epochs}"
  --data_dir "${data_dir}"
  --img_size "${IMG_SIZE}"
  --save_path "${save_path}"
  --batch_size "${batch_size}"
  --num_workers 8
  --optimizer "${optimizer}"
  --loss_type "${loss_type}"
  --loss_weights "${loss_weights}"
  --scheduler "${scheduler}"
  --lr "${lr}"
  --lr_enc "${lr_enc}"
  --encoder_ptdir "${PT_DIR}"
  --num_classes "${num_classes}"
  --input_channels "${input_channels}"
  --seed "${seed}"
  --max_grad_norm "${max_grad_norm}"
  --amp
)

case "$MODE" in
  TRAIN)
    extra=()
    [[ -n "$resume" ]] && extra+=(--resume "$resume")
    python main.py "${common[@]}" "${extra[@]}"
    ;;
  TEST)
    if [[ -z "${EVAL_PT}" ]]; then
      echo "Set EVAL_PT to a checkpoint for TEST mode."
      echo "  Skin/BUSI:  EVAL_PT=/path/to/best.pth"
      echo "  Synapse:    EVAL_PT=/path/to/best.pth  (full-volume validation on the 12 cases)"
      exit 1
    fi
    python main.py "${common[@]}" --eval --checkpoint "${EVAL_PT}" --save_test
    ;;
  *)
    echo "Invalid mode: $MODE (TRAIN|TEST)"
    exit 1
    ;;
esac

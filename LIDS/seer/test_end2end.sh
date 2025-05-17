#!/usr/bin/env bash
#-------------------------------------------------------------------------------
# Script: test_end2end.sh
# Description: Run end-to-end defense experiments over multiple datasets, attack
#              props and PSNR thresholds with configurable flags.
# Usage:
#   ./test_end2end.sh [--train True] [--seed 1234] … 
#-------------------------------------------------------------------------------

# ——— Default parameters (override via flags if you add parsing) ———
TRAIN="False"
SEED=1337
DEVICE="cuda:0"
N_COMPONENTS=17
BATCH_SIZE=64
BASE_NUM=8
INIT_FINETUNE_EPOCH=2000
FINETUNE_EPOCH=1000
FINETUNE_IMAGES_NUM=10000
INIT_ALPHA=0.5
MAX_ALPHA=0.9
DATASET_GROUP_SIZE=8
LIDS_DATASET_PATH=""
IMG_DISTANCE="MSE"
REPEAT_NUM=0
TEST_BATCHES_NUM=1000

# ——— Arrays to loop over ———
DATASETS=("Cifar10" "Cifar100")
ATK_PROPS=("blue" "bright" "dark" "green" "red" "rand_conv" "hedge" "vedge")
PSNR_THRESHOLDS=("17.0" "18.0" "19.0" "20.0" "21.0" "22.0")

# ——— Nested loops ———

DEFENSE="LIDS"

for DATASET in "${DATASETS[@]}"; do
  for ATK_PROP in "${ATK_PROPS[@]}"; do
    for PSNR_THRESHOLD in "${PSNR_THRESHOLDS[@]}"; do
      echo "=== Running: dataset=$DATASET | atk_prop=$ATK_PROP | psnr=$PSNR_THRESHOLD ==="
      python test_end2end.py \
        --train "$TRAIN" \
        --seed "$SEED" \
        --defense "$DEFENSE" \
        --device "$DEVICE" \
        --atk-prop "$ATK_PROP" \
        --n-components "$N_COMPONENTS" \
        --dataset "$DATASET" \
        --batch-size "$BATCH_SIZE" \
        --base-num "$BASE_NUM" \
        --init-finetune-epoch "$INIT_FINETUNE_EPOCH" \
        --finetune-epoch "$FINETUNE_EPOCH" \
        --finetune-images-num "$FINETUNE_IMAGES_NUM" \
        --psnr-threshold "$PSNR_THRESHOLD" \
        --init-alpha "$INIT_ALPHA" \
        --max-alpha "$MAX_ALPHA" \
        --dataset-group-size "$DATASET_GROUP_SIZE" \
        --lids-dataset-path "$LIDS_DATASET_PATH" \
        --img-distance "$IMG_DISTANCE" \
        --repeat-num "$REPEAT_NUM" \
        --test-batches-num "$TEST_BATCHES_NUM"
      echo
    done
  done
done

DEFENSE="LIDS-A"

for DATASET in "${DATASETS[@]}"; do
  for ATK_PROP in "${ATK_PROPS[@]}"; do
    for PSNR_THRESHOLD in "${PSNR_THRESHOLDS[@]}"; do
      echo "=== Running: dataset=$DATASET | atk_prop=$ATK_PROP | psnr=$PSNR_THRESHOLD ==="
      python test_end2end.py \
        --train "$TRAIN" \
        --seed "$SEED" \
        --defense "$DEFENSE" \
        --device "$DEVICE" \
        --atk-prop "$ATK_PROP" \
        --n-components "$N_COMPONENTS" \
        --dataset "$DATASET" \
        --batch-size "$BATCH_SIZE" \
        --base-num "$BASE_NUM" \
        --init-finetune-epoch "$INIT_FINETUNE_EPOCH" \
        --finetune-epoch "$FINETUNE_EPOCH" \
        --finetune-images-num "$FINETUNE_IMAGES_NUM" \
        --psnr-threshold "$PSNR_THRESHOLD" \
        --init-alpha "$INIT_ALPHA" \
        --max-alpha "$MAX_ALPHA" \
        --dataset-group-size "$DATASET_GROUP_SIZE" \
        --lids-dataset-path "$LIDS_DATASET_PATH" \
        --img-distance "$IMG_DISTANCE" \
        --repeat-num "$REPEAT_NUM" \
        --test-batches-num "$TEST_BATCHES_NUM"
      echo
    done
  done
done

DEFENSE="NO-DEFENSE"

for DATASET in "${DATASETS[@]}"; do
  for ATK_PROP in "${ATK_PROPS[@]}"; do
    echo "=== Running: dataset=$DATASET | atk_prop=$ATK_PROP ==="
    python test_end2end.py \
      --train "$TRAIN" \
      --seed "$SEED" \
      --defense "$DEFENSE" \
      --device "$DEVICE" \
      --atk-prop "$ATK_PROP" \
      --n-components "$N_COMPONENTS" \
      --dataset "$DATASET" \
      --batch-size "$BATCH_SIZE" \
      --base-num "$BASE_NUM" \
      --init-finetune-epoch "$INIT_FINETUNE_EPOCH" \
      --finetune-epoch "$FINETUNE_EPOCH" \
      --finetune-images-num "$FINETUNE_IMAGES_NUM" \
      --init-alpha "$INIT_ALPHA" \
      --max-alpha "$MAX_ALPHA" \
      --dataset-group-size "$DATASET_GROUP_SIZE" \
      --lids-dataset-path "$LIDS_DATASET_PATH" \
      --img-distance "$IMG_DISTANCE" \
      --repeat-num "$REPEAT_NUM" \
      --test-batches-num "$TEST_BATCHES_NUM"
    echo
  done
done 
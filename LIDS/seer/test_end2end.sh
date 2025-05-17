#!/usr/bin/env bash
#-------------------------------------------------------------------------------
# Script: test_end2end.sh
# Description: Wrapper to launch defense experiment script with configurable flags.
# Usage:
#   ./test_end2end.sh --model my_model.params --train True --dataset Cifar100 ...
#-------------------------------------------------------------------------------

# Default parameters (override via flags)
TRAIN="False"
SEED=1337
DEFENSE="LIDS"
DEVICE="cuda:0"
ATK_PROP="bright"
N_COMPONENTS=17
DATASET="Cifar10"
RANDOM_LOADER="False"
BATCH_SIZE=64
BASE_NUM=8
INIT_FINETUNE_EPOCH=2000
FINETUNE_EPOCH=1000
FINETUNE_IMAGES_NUM=10000
PSNR_THRESHOLD=18.0
INIT_ALPHA=0.5
MAX_ALPHA=0.9
DATASET_GROUP_SIZE=8
LIDS_DATASET_PATH=""
IMG_DISTANCE="MSE"
REPEAT_NUM=0
TEST_BATCHES_NUM=1000

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)               TRAIN="$2"; shift 2 ;; 
    --seed)                SEED="$2"; shift 2 ;; 
    --defense)             DEFENSE="$2"; shift 2 ;; 
    --device)              DEVICE="$2"; shift 2 ;; 
    --atk-prop)            ATK_PROP="$2"; shift 2 ;; 
    --n-components)        N_COMPONENTS="$2"; shift 2 ;; 
    --dataset)             DATASET="$2"; shift 2 ;; 
    --random-loader)       RANDOM_LOADER="$2"; shift 2 ;; 
    --batch-size)          BATCH_SIZE="$2"; shift 2 ;; 
    --base-num)            BASE_NUM="$2"; shift 2 ;; 
    --init-finetune-epoch) INIT_FINETUNE_EPOCH="$2"; shift 2 ;; 
    --finetune-epoch)      FINETUNE_EPOCH="$2"; shift 2 ;; 
    --finetune-images-num) FINETUNE_IMAGES_NUM="$2"; shift 2 ;; 
    --psnr-threshold)      PSNR_THRESHOLD="$2"; shift 2 ;; 
    --init-alpha)          INIT_ALPHA="$2"; shift 2 ;; 
    --max-alpha)           MAX_ALPHA="$2"; shift 2 ;; 
    --dataset-group-size)  DATASET_GROUP_SIZE="$2"; shift 2 ;; 
    --lids-dataset-path)   LIDS_DATASET_PATH="$2"; shift 2 ;; 
    --img-distance)        IMG_DISTANCE="$2"; shift 2 ;; 
    --repeat-num)          REPEAT_NUM="$2"; shift 2 ;; 
    --test-batches-num)    TEST_BATCHES_NUM="$2"; shift 2 ;; 
    *) echo "Unknown option: $1"; exit 1 ;; 
  esac
done


# Show final configuration
echo "Configuration:"
echo "  TRAIN=$TRAIN"
echo "  SEED=$SEED"
echo "  DEFENSE=$DEFENSE"
echo "  DEVICE=$DEVICE"
echo "  ATK_PROP=$ATK_PROP"
echo "  N_COMPONENTS=$N_COMPONENTS"
echo "  DATASET=$DATASET"
echo "  RANDOM_LOADER=$RANDOM_LOADER"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BASE_NUM=$BASE_NUM"
echo "  INIT_FINETUNE_EPOCH=${INIT_FINETUNE_EPOCH}"
echo "  FINETUNE_EPOCH=$FINETUNE_EPOCH"
echo "  FINETUNE_IMAGES_NUM=$FINETUNE_IMAGES_NUM"
echo "  PSNR_THRESHOLD=$PSNR_THRESHOLD"
echo "  INIT_ALPHA=$INIT_ALPHA"
echo "  MAX_ALPHA=$MAX_ALPHA"
echo "  DATASET_GROUP_SIZE=$DATASET_GROUP_SIZE"
echo "  LIDS_DATASET_PATH=$LIDS_DATASET_PATH"
echo "  IMG_DISTANCE=$IMG_DISTANCE"
echo "  REPEAT_NUM=$REPEAT_NUM"
echo "  TEST_BATCHES_NUM=$TEST_BATCHES_NUM"

echo
# Execute Python experiment
python test_end2end.py \
  --train "$TRAIN" \
  --seed $SEED \
  --defense "$DEFENSE" \
  --device "$DEVICE" \
  --atk-prop "$ATK_PROP" \
  --n-components $N_COMPONENTS \
  --dataset "$DATASET" \
  --random-loader $RANDOM_LOADER \
  --batch-size $BATCH_SIZE \
  --base-num $BASE_NUM \
  --finetune-epoch $FINETUNE_EPOCH \
  --init-finetune-epoch ${INIT_FINETUNE_EPOCH} \
  --finetune-images-num $FINETUNE_IMAGES_NUM \
  --psnr-threshold $PSNR_THRESHOLD \
  --init-alpha $INIT_ALPHA \
  --max-alpha $MAX_ALPHA \
  --dataset-group-size $DATASET_GROUP_SIZE \
  --lids-dataset-path "$LIDS_DATASET_PATH" \
  --img-distance "$IMG_DISTANCE" \
  --repeat-num $REPEAT_NUM \
  --test-batches-num $TEST_BATCHES_NUM

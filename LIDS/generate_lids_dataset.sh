#!/usr/bin/env bash
#-------------------------------------------------------------------------------
# Script: generate_lids_dataset.sh
# Description: Run the LIDS dataset generation script with configurable parameters.
# Usage:
#   TRAIN=True SEED=1234 DEVICE=cuda:0 ./generate_lids_dataset.sh
#-------------------------------------------------------------------------------

# Default parameters (can be overridden via environment variables)
TRAIN=${TRAIN:-"False"}               # Use training set if True, otherwise test set
SEED=${SEED:-1337}                     # Random seed for reproducibility
DEVICE=${DEVICE:-"cuda:0"}           # Computation device
ATK_PROP=${ATK_PROP:-"bright"}       # Attack property
N_COMPONENTS=${N_COMPONENTS:-17}       # Number of PCA components
DATASET=${DATASET:-"Cifar10"}        # Dataset selection: Cifar10 or Cifar100
BATCH_SIZE=${BATCH_SIZE:-64}           # Batch size
BASE_NUM=${BASE_NUM:-8}                # Base batch size for SMOTE groups
INIT_FINETUNE_EPOCH=${INIT_FINETUNE_EPOCH:-2000} # Number of initial finetuning epochs
FINETUNE_EPOCH=${FINETUNE_EPOCH:-1000} # Number of finetuning epochs
FINETUNE_IMAGES_NUM=${FINETUNE_IMAGES_NUM:-10000} # Images per finetune group
PSNR_THRESHOLD=${PSNR_THRESHOLD:-18.0} # PSNR threshold for sample selection
INIT_ALPHA=${INIT_ALPHA:-0.5}          # Initial interpolation weight
MAX_ALPHA=${MAX_ALPHA:-0.9}            # Maximum interpolation weight
AUGMENT=${AUGMENT:-"False"}           # Apply contrast augmentations if True
IMG_DISTANCE=${IMG_DISTANCE:-"MSE"}  # Distance metric for image comparison
DENORMALIZE=${DENORMALIZE:-"False"} # Denormalize the lids dataset or not
# Print configuration
echo "Configuration:"
echo "  TRAIN=${TRAIN}"
echo "  SEED=${SEED}"
echo "  DEVICE=${DEVICE}"
echo "  ATK_PROP=${ATK_PROP}"
echo "  N_COMPONENTS=${N_COMPONENTS}"
echo "  DATASET=${DATASET}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  BASE_NUM=${BASE_NUM}"
echo "  INIT_FINETUNE_EPOCH=${INIT_FINETUNE_EPOCH}"
echo "  FINETUNE_EPOCH=${FINETUNE_EPOCH}"
echo "  FINETUNE_IMAGES_NUM=${FINETUNE_IMAGES_NUM}"
echo "  PSNR_THRESHOLD=${PSNR_THRESHOLD}"
echo "  INIT_ALPHA=${INIT_ALPHA}"
echo "  MAX_ALPHA=${MAX_ALPHA}"
echo "  AUGMENT=${AUGMENT}"
echo "  IMG_DISTANCE=${IMG_DISTANCE}"
echo "  DENORMALIZE=${DENORMALIZE}"

echo
# Execute Python script
python generate.py \
  --train "${TRAIN}" \
  --seed ${SEED} \
  --device "${DEVICE}" \
  --atk-prop "${ATK_PROP}" \
  --n-components ${N_COMPONENTS} \
  --dataset "${DATASET}" \
  --batch-size ${BATCH_SIZE} \
  --base-num ${BASE_NUM} \
  --init-finetune-epoch ${INIT_FINETUNE_EPOCH} \
  --finetune-epoch ${FINETUNE_EPOCH} \
  --finetune-images-num ${FINETUNE_IMAGES_NUM} \
  --psnr-threshold ${PSNR_THRESHOLD} \
  --init-alpha ${INIT_ALPHA} \
  --max-alpha ${MAX_ALPHA} \
  --augment "${AUGMENT}" \
  --img_distance "${IMG_DISTANCE}" \
  --denormalize "${DENORMALIZE}"

# End of script

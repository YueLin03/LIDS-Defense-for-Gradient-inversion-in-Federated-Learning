import argparse
import os
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T

from lids_group_autoencoder import generate_dataset
from lids_dataset import LIDS_Dataset


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dataset generation for LIDS framework")
    parser.add_argument('--train', type=str, choices=['False', 'True'], default='False',
                        help='Use training set if True, otherwise test set')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Computation device')
    parser.add_argument('--atk-prop', type=str, default="bright",
                        help='Attack property')
    parser.add_argument('--n-components', type=int, default=17,
                        help='Number of components for interpolation')
    parser.add_argument('--dataset', type=str, default="Cifar10",
                        help='Dataset selection: Cifar10 or Cifar100')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--base-num', type=int, default=8,
                        help='Base batch size for SMOTE groups')
    parser.add_argument('--finetune-epoch', type=int, default=1000,
                        help='Number of finetuning epochs')
    parser.add_argument('--finetune-images-num', type=int, default=10000,
                        help='Number of images per finetune group')
    parser.add_argument('--psnr-threshold', type=float, default=18,
                        help='PSNR threshold for sample selection')
    parser.add_argument('--init-alpha', type=float, default=0.5,
                        help='Initial interpolation weight')
    parser.add_argument('--max-alpha', type=float, default=0.9,
                        help='Maximum interpolation weight')
    parser.add_argument('--augment', type=str, choices=['False', 'True'], default='False',
                        help='Apply contrast augmentations if True')
    parser.add_argument('--img_distance', type=str, default="MSE",
                        help='Distance metric for image comparison')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define transformation: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
            std=(0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
        )
    ])

    # Load CIFAR dataset based on arguments
    is_train = (args.train == 'True')
    ds_name = args.dataset.lower()
    if ds_name == 'cifar10':
        split = 'Trainset' if is_train else 'Testset'
        print(f"Loading CIFAR-10 {split}")
        target_dataset = torchvision.datasets.CIFAR10(
            root='../data', train=is_train, download=True, transform=transform
        )
    elif ds_name == 'cifar100':
        split = 'Trainset' if is_train else 'Testset'
        print(f"Loading CIFAR-100 {split}")
        target_dataset = torchvision.datasets.CIFAR100(
            root='../data', train=is_train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Prepare dataset directories
    dataset_dir = (
        f"./lids_datasets/{args.dataset}/train_{args.train}/"
        f"denorm_{args.denormalize}/{args.img_distance}/"
        f"finetune_i{args.finetune_init_epoch}_n{args.finetune_images_num}_"
        f"e{args.finetune_epoch}/psnr_thresh{args.psnr_threshold}"
    )
    lids_A_path = os.path.join(dataset_dir, 'aug_False.dst')
    lids_path = os.path.join(dataset_dir, 'aug_True.dst')

    # Generate or load augmented dataset A (without contrast augmentation)
    if not os.path.exists(lids_A_path):
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Original dataset size: {len(target_dataset)}")
        new_ds, smote_base, smote_neighbor, smote_alpha = generate_dataset(args, target_dataset)
        print(f"Generated samples: {len(new_ds)}")

        # Combine original and generated samples
        origin_imgs = torch.stack([target_dataset[i][0] for i in range(len(target_dataset))])
        origin_lbls = torch.tensor([target_dataset[i][1] for i in range(len(target_dataset))])
        gen_imgs, gen_lbls = new_ds.tensors

        all_imgs = torch.cat([origin_imgs, gen_imgs], dim=0)
        all_lbls = torch.cat([origin_lbls, gen_lbls], dim=0)

        ae_dataset = LIDS_Dataset(
            all_imgs, all_lbls, smote_base, smote_neighbor, smote_alpha, len(target_dataset)
        )
        print(f"Combined dataset size: {len(ae_dataset)}")

        # Save Dataset A
        torch.save(ae_dataset, lids_A_path)
        print(f"Dataset A saved at: {lids_A_path}")
    else:
        ae_dataset = torch.load(lids_A_path)
        print(f"Loaded existing Dataset A from: {lids_A_path}")

    # Apply contrast-based augmentations if requested
    all_data = ae_dataset.data
    origin_len = ae_dataset.origin_datalen
    mean = torch.tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822]).view(3,1,1)
    std = torch.tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324]).view(3,1,1)

    processed = []
    for idx in range(all_data.size(0)):
        img = all_data[idx] * std + mean       # Denormalize
        img = torch.clamp(img, 0, 1)
        if idx >= origin_len and args.augment == 'True':
            mod = (idx - origin_len) % 7
            # Contrast and color adjustments based on group index
            if mod == 0:
                img = F.adjust_contrast(img, random.uniform(1.5, 2.0))
            elif mod == 1:
                img = F.adjust_brightness(img, random.uniform(1.1, 1.3))
                img = F.adjust_contrast(img, random.uniform(1.3, 1.5))
            elif mod == 2:
                img = F.adjust_contrast(img, random.uniform(1.3, 1.5))
                img = F.adjust_saturation(img, random.uniform(1.4, 1.6))
                img = F.adjust_brightness(img, random.uniform(0.8, 1.1))
            elif mod == 3:
                img = F.adjust_hue(img, random.uniform(0.03, 0.08))
                img = F.adjust_contrast(img, random.uniform(1.1, 1.3))
            elif mod == 4:
                img = T.functional.gaussian_blur(img, kernel_size=3)
                img = F.adjust_saturation(img, random.uniform(1.3, 1.6))
            elif mod == 5:
                img = T.functional.gaussian_blur(img, kernel_size=3)
                img = T.functional.adjust_sharpness(img, random.uniform(1.5, 2.2))
            else:
                img = F.adjust_contrast(img, random.uniform(1.1, 1.3))
                hue_shift = random.choice([-1, 1]) * random.uniform(0.03, 0.07)
                img = F.adjust_hue(img, hue_shift)
            img = torch.clamp(img, 0, 1)
        # Re-normalize test set images
        if not is_train:
            img = (img - mean) / std
        processed.append(img)

    # Create new dataset with processed images
    new_ds = ae_dataset
    new_ds.data = torch.stack(processed)
    print(f"Processed dataset size: {len(new_ds)}")

    # Save Dataset B (with augmentations)
    if not os.path.exists(lids_path):
        torch.save(new_ds, lids_path)
        print(f"Dataset B saved at: {lids_path}")
    else:
        print("Dataset B already exists.")


if __name__ == '__main__':
    main()

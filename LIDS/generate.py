import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import adjust_contrast
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter, GaussianBlur, RandomApply
import random
import numpy as np
import os
import argparse

from lids_group_autoencoder import generate_dataset
from lids_dataset import LIDS_Dataset,SimpleSampler
# 固定隨機種子
seed = 1337
np.random.seed(seed)

parser = argparse.ArgumentParser(description="Example parser", add_help=False, exit_on_error=False)
parser.add_argument('--train', type=str,choices=['False', 'True'], default='False')
parser.add_argument('--seed', type=int, default=seed)
parser.add_argument('--name', type=str, default="default")
parser.add_argument('--device', type=str, help='device', default="cuda:0")
parser.add_argument('--scoring', type=str,default="generate_dataset")
parser.add_argument('--atk-prop', type=str, default="bright")
parser.add_argument('--def-prop', type=str, default="bright")
parser.add_argument('--n-components', type=int, default=17)
parser.add_argument('--dataset', type=str, default="Cifar10")
parser.add_argument('--model', type=str)
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--base-num', type=int, default=8, help='Batch size for training')
parser.add_argument('--finetune-epoch', type=int, default=1000)
parser.add_argument('--finetune-images-num', type=int, default=10000)
parser.add_argument('--psnr-threshold', type=float, default=18, help='PSNR threshold for sample selection')
parser.add_argument('--init-alpha', type=float, default=0.05, help='Initial alpha value for interpolation')
parser.add_argument('--max-alpha', type=float, default=0.95, help='Maximum alpha value for interpolation')
parser.add_argument('--augment', type=str, default="Norm")
parser.add_argument('--img_distance', type=str, default="MSE")
args = parser.parse_args()
print(args)

if args.augment == "Paper":
    transform_dst = transforms.Compose(
    [transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((-5,5), fill=255),
            transforms.RandomRotation((85,95), fill=255),
            transforms.RandomRotation((175,185), fill=255),
            transforms.RandomRotation((-95,-85), fill=255)
        ]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
elif args.augment == "Norm":
    transform_dst = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
else:
    print(f"No such augment method: {args.augment}")
    raise ValueError()

contrast_factor = 2
sharpness_factor = 1
if args.dataset == "Cifar10":
    if args.train == "True":
        print("Dataset = Cifar10 Trainset")
        dataset_dir = f'./datasets/{args.dataset}/trainset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}'
        new_dataset_dir = f'./datasets/{args.dataset}/trainset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}/contrast_{contrast_factor}'
        eval_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_dst)
    else:
        dataset_dir = f'./datasets/{args.dataset}/testset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}'
        print("Dataset = Cifar10 Testset")
        new_dataset_dir = f'./datasets/{args.dataset}/testset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}/contrast_{contrast_factor}'
        eval_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_dst)
elif args.dataset == "Cifar100":
    if args.train == "True":
        print("Dataset = Cifar100 Trainset")
        dataset_dir = f'./datasets/{args.dataset}/trainset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}'
        new_dataset_dir = f'./datasets/{args.dataset}/trainset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}/contrast_{contrast_factor}'
        eval_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_dst)
    else:
        dataset_dir = f'./datasets/{args.dataset}/testset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}'
        new_dataset_dir = f'./datasets/{args.dataset}/testset/init_2000/{args.augment}/{args.img_distance}/paper/init-{args.init_alpha}-max-{args.max_alpha}/contrast_{contrast_factor}'
        print("Dataset = Cifar100 Testset")
        eval_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_dst)
print(args)
print(f"model = {args.model}")
device = args.device
print(f"dataset_dir = {dataset_dir}")
print(f"new_dataset_dir = {new_dataset_dir}")
dataset_path = dataset_dir + f'/finetune{args.finetune_epoch}epoch_psnr_thresh{args.psnr_threshold}finetune_images_num{args.finetune_images_num}.dst'
if not os.path.exists(dataset_path):
    # 呼叫 SMOTE 分組函數
    print(f"origin dataset size: {len(eval_dataset)}")
    new_eval_dataset, smote_to_base, smote_to_neighbor, smote_to_alpha = generate_dataset(args, eval_dataset)
    print(f"len new_eval_dataset = {len(new_eval_dataset)}")
    print(f"len smote_to_base = {len(smote_to_base)}")
    print(f"len smote_to_neighbor = {len(smote_to_neighbor)}")
    print(f"len smote_to_alpha = {len(smote_to_alpha)}")


    # 整合數據
    origin_data = torch.stack([eval_dataset[i][0] for i in range(len(eval_dataset))])
    origin_labels = torch.tensor([eval_dataset[i][1] for i in range(len(eval_dataset))])
    generated_data, generated_labels = new_eval_dataset.tensors

    all_data = torch.cat([origin_data, generated_data], dim=0)
    all_labels = torch.cat([origin_labels, generated_labels], dim=0)

    ae_dataset = LIDS_Dataset(all_data, all_labels, smote_to_base, smote_to_neighbor, smote_to_alpha, len(eval_dataset))

    # 儲存資料集
    print(f"Gen dataset size: {len(ae_dataset)}")

    os.makedirs(dataset_dir, exist_ok=True)
    torch.save(ae_dataset,dataset_path)
    print(f"Dataset created at: {dataset_path}")
else: 
    ae_dataset = torch.load(dataset_path)
print("##Save contrast dataset")
all_data = ae_dataset.data
print(f"all_data shape = {all_data.shape}")#all_data shape = torch.Size([80000, 3, 32, 32])
print(f"origin_datalen shape = {ae_dataset.origin_datalen}")

mean = torch.tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822]).view(3, 1, 1)
std = torch.tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324]).view(3, 1, 1)

batch_size, C, H, W = all_data.shape
processed_images = []

#print(X[0])
for i in range(batch_size):
    img = all_data[i]
    img = img * std + mean  
    img = torch.clamp(img, 0, 1)  
    if i in range(ae_dataset.origin_datalen,batch_size):
        i_type = (i-ae_dataset.origin_datalen) % 7
        if (i_type == 0):
            img = adjust_contrast(img, random.uniform(1.5, 2))  # 調整對比度 
        elif (i_type == 1):
            img = F.adjust_brightness(img, random.uniform(1.1, 1.3))
            img = F.adjust_contrast(img, random.uniform(1.3, 1.5))
        elif (i_type == 2):
            img = F.adjust_contrast(img, random.uniform(1.3, 1.5))
            img = F.adjust_saturation(img, random.uniform(1.4, 1.6))
            img = F.adjust_brightness(img, random.uniform(0.8, 1.1))
        elif (i_type == 3):
            # 色相微調偏紅（讓圖像偏暖色調）
            img = F.adjust_hue(img, random.uniform(0.03, 0.08))
            img = F.adjust_contrast(img, random.uniform(1.1, 1.3))       
        elif (i_type == 4):
            # 小幅 Gaussian 模糊 + 提升飽和度，視覺更柔和鮮明
            img = T.functional.gaussian_blur(img, kernel_size=3)
            img = F.adjust_saturation(img, random.uniform(1.3, 1.6))
        elif (i_type == 5):
            img = T.functional.gaussian_blur(img, kernel_size=3)  # 先做平滑以避免過度鋭化
            img = T.functional.adjust_sharpness(img, random.uniform(1.5, 2.2))  # 調整鋭化度
        elif (i_type == 6):
            # 對比 & 色相雙重小調整（讓圖像略偏冷/暖色，但不破壞）
            img = F.adjust_contrast(img, random.uniform(1.1, 1.3))
            hue_shift = random.choice([-1, 1]) * random.uniform(0.03, 0.07)
            img = F.adjust_hue(img, hue_shift)
    
        img = torch.clamp(img, 0, 1)  # 確保數值仍在 [0,1] 之間
    if args.train == "False": #如果是testset要重新標準化
        img = (img - mean) / std  #重新標準化
            
    processed_images.append(img)
X = torch.stack(processed_images)  

new_dataset = ae_dataset
new_dataset.data = X
# 儲存資料集
print(f"Gen dataset size: {len(new_dataset)}")
#print(f"First item: {new_dataset[0]}")
new_dataset_path = new_dataset_dir + f'/finetune{args.finetune_epoch}epoch_psnr_thresh{args.psnr_threshold}finetune_images_num{args.finetune_images_num}.dst'
os.makedirs(new_dataset_dir, exist_ok=True)
if not os.path.exists(new_dataset_path):
    torch.save(new_dataset,new_dataset_path)
    print(f"Dataset created at: {new_dataset_path}")
else :
    print("Already exists!!!")


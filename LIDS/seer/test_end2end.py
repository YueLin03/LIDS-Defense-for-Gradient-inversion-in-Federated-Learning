import os
import argparse
import time
import csv
import pickle
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from train import *
from data import *
from model import *
from lids_dataset import LIDS_Dataset
from dataloader import SimpleSampler
from simple_breach import run_metrics
from parameters import *
# -------------------------------------
# Argument Parser
# -------------------------------------
parser = argparse.ArgumentParser(description='sampler defense')
parser.add_argument('--train', type=str, choices=['False','True'], default='False')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--defense', type=str, choices=['default','random','LIDS','LIDS-A'], default='default')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--atk-prop', type=str, default='bright')
parser.add_argument('--n-components', type=int, default=17)
parser.add_argument('--dataset', type=str, choices=['Cifar10','Cifar100'], default='Cifar10')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--random-loader', type=bool, default=False)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--base-num', type=int, default=8)
parser.add_argument('--finetune-epoch', type=int, default=1000)
parser.add_argument('--finetune-images-num', type=int, default=10000)
parser.add_argument('--psnr-threshold', type=float, default=18.0)
parser.add_argument('--init-alpha', type=float, default=0.5)
parser.add_argument('--max-alpha', type=float, default=0.9)
parser.add_argument('--dataset-group-size', type=int, default=8)
parser.add_argument('--lids-dataset-path', type=str, default=None)
parser.add_argument('--img-distance', type=str, default='MSE')
parser.add_argument('--repeat-num', type=int, default=0)
parser.add_argument('--test-batches-num', type=int, default=10)
args = parser.parse_args()
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
print(args)

# -------------------------------------
# Build LIDS Dataset Paths
# -------------------------------------
base_dataset_dir = (
    f"./lids_datasets/{args.dataset}/train_{args.train}/"
    f"denorm_{args.denormalize}/{args.img_distance}/"
    f"finetune_i{args.init_finetune_epoch}_n{args.finetune_images_num}_"
    f"e{args.finetune_epoch}/psnr_thresh{args.psnr_threshold}"
)

lids_A_path = os.path.join(base_dataset_dir, 'aug_False.dst')
lids_path   = os.path.join(base_dataset_dir, 'aug_True.dst')

if args.defense in ['LIDS','LIDS-A'] and args.lids_dataset_path is None:
    args.lids_dataset_path = lids_A_path if args.defense=='LIDS-A' else lids_path

# -------------------------------------
# Dataset & Transforms
# -------------------------------------
def add_red_tint(img):
    if not torch.is_tensor(img):
        raise TypeError("Input should be a PyTorch Tensor")
    img = img.clone(); img[0] += 0.2
    return torch.clamp(img,0,1)

def datasets_Cifar10():
    transform_train = transforms.Compose(
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
    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    transform_red_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(add_red_tint),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    red_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_red_test)

    return trainset,testset,red_dataset
def datasets_Cifar100():
    transform_train = transforms.Compose(
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
    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    transform_red_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(add_red_tint),
     transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])

    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    red_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_red_test)

    return trainset,testset,red_dataset
def create_subset(dataset):
    num_samples = int(len(dataset) * 0.2)

    # Randomly select indices for the new dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Create the subset
    subset_dataset = Subset(dataset, indices)

    return subset_dataset

# Select dataset
if args.dataset=='Cifar10':
    trainset,testset,red_dataset = datasets_Cifar10()
elif args.dataset=='Cifar100':
    trainset,testset,red_dataset = datasets_Cifar100()

# Evaluation dataset
eval_dataset = testset if args.defense!='red_dataset' else red_dataset



args.model = f"../weights/B64C1{args.atk_prop}{args.dataset}Epoch1000.params"
checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
print(f"model{args.model}")
device = args.device

if (args.dataset =="Cifar10"):
    public_model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], "ReLU", num_classes=10)
elif (args.dataset =="Cifar100"):
    public_model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], "ReLU", num_classes=100)


# select specific parameter from public model
par_sel=ParamSelector(public_model,8400,0.001,sparse_grad=False,seed=98)
grad_ex=GradientExtractor(public_model,par_sel)
reconstructor_id=num_params(grad_ex)
reconstructor_od=3*32*32
disaggregator=torch.nn.Identity()
reconstructor=torch.nn.Linear(reconstructor_id,reconstructor_od,bias=True)

public_model.load_state_dict(checkpoint['public_model_state_dict'])
# public_model
disaggregator.load_state_dict(checkpoint['disaggregator_state_dict'])
# disaggregator
reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
 
def denormalize_cifar10(tensor):
    """
    Denormalize CIFAR10 images from tensor format to [0,1] range
    """
    mean = torch.tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822]).view(3, 1, 1)
    std = torch.tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324]).view(3, 1, 1)
    
    return tensor * std + mean

def get_top_k_indices(psnr_scores, k=4):
    """Get indices of top k PSNR scores"""
    return np.argsort(psnr_scores)[-k:]




if args.defense == "random":
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=2,shuffle=False,sampler = torch.utils.data.RandomSampler(eval_dataset, replacement=False, num_samples=int(1e10))) 
elif args.defense in ['LIDS','LIDS-A']:
    ae_dataset = torch.load(args.lids_dataset_path)
    print(f"Loaded dataset type: {type(ae_dataset)}")
    print(f"Testset len: {len(ae_dataset)}")
    smote_to_base = {}
    smote_to_neighbor = {}
    smote_to_alpha = {}
    for smote, base in ae_dataset.smote_to_base.items():
        smote_to_base[smote+ae_dataset.origin_datalen] = base
    for smote, neighbor in ae_dataset.smote_to_neighbor.items():
        smote_to_neighbor[smote+ae_dataset.origin_datalen] = neighbor    
    for smote, alpha in ae_dataset.smote_to_alpha.items():
        smote_to_alpha[smote+ae_dataset.origin_datalen] = alpha    
    sampler = SimpleSampler(ae_dataset.origin_datalen, len(ae_dataset),args.dataset_group_size,int(args.batch_size // args.base_num),args.repeat_num,ae_dataset.base_to_smote)
    batches_image_indices = sampler.final_indices
    batch_indices = []

    num_batches = len(batches_image_indices) // args.base_num + 1
    batch_indices = [[] for _ in range(num_batches)] 
    for batch_idx,batch in enumerate (batches_image_indices):
        for idx in batch:
            batch_indices[int(batch_idx//args.base_num)].append(idx)
    print(batch_indices)
    loader = DataLoader(
        ae_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        sampler=sampler,
    )
else:
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
# -------------------------------------
# Output & Test Data Paths
# -------------------------------------
# Images output
output_folder = os.path.join('images', args.dataset, args.defense, args.atk_prop)
if args.defense in ['LIDS','LIDS-A']:
    output_folder = os.path.join(
        output_folder,
        args.img_distance,
        f"base_num_{args.base_num}",
        f"finetune_i{args.init_finetune_epoch}_n{args.finetune_images_num}_e{args.finetune_epoch}",
        f"init-{args.init_alpha}-max-{args.max_alpha}"
    )
os.makedirs(output_folder, exist_ok=True)

# Test data pickle output
test_data_output_dir = os.path.join('test_data', args.dataset, args.defense, args.atk_prop)
if args.defense in ['LIDS','LIDS-A']:
    test_data_output_dir = os.path.join(
        test_data_output_dir,
        f"base_num_{args.base_num}",
        f"finetune_i{args.init_finetune_epoch}_n{args.finetune_images_num}_e{args.finetune_epoch}",
        f"init-{args.init_alpha}-max-{args.max_alpha}"
    )
os.makedirs(test_data_output_dir, exist_ok=True)

# CSV output
csv_output_dir = os.path.join('csv_file', args.dataset, args.defense, args.atk_prop)
if args.defense in ['LIDS','LIDS-A']:
    csv_output_dir = os.path.join(
        'csv_file',
        f"base_num_{args.base_num}",
        f"finetune_i{args.init_finetune_epoch}_n{args.finetune_images_num}_e{args.finetune_epoch}",
        f"init-{args.init_alpha}-max-{args.max_alpha}"
    )
os.makedirs(csv_output_dir, exist_ok=True)

ssims = []
psnrs = []
base_ssims = []
base_psnrs = []
class Args:
    def __init__(self):
        self.device = args.device
        self.neptune = False
        self.attack_cfg = 'modern'
        self.case_cfg = f'Cifar10_sanity_check'
        if (args.dataset == "Cifar100"):
            self.case_cfg = f'Cifar100_sanity_check'
args_br = Args()
br_reconstr = BreachingReconstruction(args_br, public_model, torch.nn.CrossEntropyLoss(), 64, 1, dtype=torch.float32)
dm = torch.as_tensor(br_reconstr.user.dataloader.dataset.mean, **br_reconstr.setup)[None, :, None, None].to("cpu")
ds = torch.as_tensor(br_reconstr.user.dataloader.dataset.std, **br_reconstr.setup)[None, :, None, None].to("cpu")

def save_image(image, file_path):
    # Denormalize tensor to range [0, 1]
    tensor_image = torch.tensor(image)
    image_array = torch.clamp(tensor_image, 0, 1).cpu().numpy()
    # Create a figure and axis to plot the image
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    ax.axis('off')  # Remove axes for a clean image
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to avoid memory issues
print(f"attack prop: {args.atk_prop}, defense prop: {args.def_prop}")

TARGET_indices, TARGET_class = [], []
ssims, psnrs, lpips_list = [], [], []
metrics = []


repeat_num = args.batch_size // args.base_num

for i, (X, y) in enumerate(loader):
    if X.shape[0] != args.batch_size: 
        continue  
    if (i >= args.test_batches_num):
        break
    print(f"Processing Batch {i}...")

    
    bdW, _ = grad_ex(X, y, torch.nn.CrossEntropyLoss(reduction='none'), flat_cat=True, single_grad=True, testing=True)
    disaggregator_o = disaggregator(bdW)
    reconstructor_o = reconstructor(disaggregator_o).reshape(-1, *(3, 32, 32))
    
    # 計算屬性分數
    scores = property_scores(X, args.atk_prop, y)
    batch_indices = []
    

    if args.defense == "LIDS":
        batch_indices = list(iter(loader.sampler))[i*args.batch_size:(i+1)*args.batch_size]
        print(f"batch_idx {i} batch_indices: {batch_indices}")
        neiglhbor_num = int(args.batch_size // args.base_num)-1
        batch_test_idx = batch_indices
        # change smote to neighbor imgs
        test_idx = [idx if idx < ae_dataset.origin_datalen else smote_to_neighbor[idx] for idx in batch_test_idx]
        print(f"Target idx indices: {test_idx}")
        test_images = []
        for idx in test_idx:
            test_images.append(ae_dataset.data[idx])
        if len(test_images) != 0:
            test_tensor = torch.stack(test_images) #[64,3,32,32]
        test_tensor = torch.clamp(test_tensor.to(ds.device) * ds.view(1,-1,1,1) + dm.view(1,-1,1,1), 0, 1)
        reconstructor_o = torch.clamp(reconstructor_o.to(ds.device) * ds + dm, 0, 1)
        metrix = run_metrics(reconstructor_o, test_tensor, order_batch=True, log=False)

        target_idx = test_idx[metrix['selector']]
        target_smote_idx = batch_test_idx[metrix['selector']]
        if (target_smote_idx >= ae_dataset.origin_datalen):
            target_base_idx = smote_to_base[target_smote_idx]
        else:
            target_base_idx = target_smote_idx
        print(f"target base idx: {target_base_idx}")
        base_tensor = torch.clamp(ae_dataset.data[target_base_idx].to(ds.device) * ds + dm, 0, 1)
        
        target_tensor = test_tensor[metrix["selector"]]
        PSNR = metrix['max_psnr']
        SSIM = metrix['max_ssim']
        LPIPS = metrix['max_lpips']
        psnrs.append(PSNR)
        ssims.append(SSIM)
        lpips_list.append(LPIPS)
        target_class = ae_dataset.labels[target_idx]
        TARGET_indices.append(target_idx)
        TARGET_class.append(target_class)
        print(f"target_idx = {target_idx}")
        print(f"PSNR = {PSNR}")
        print(f"SSIM = {SSIM}")
        print(f"LPIPS = {LPIPS}")
        metrics.append({
            "tar_denorm": target_tensor,
            "rec_denorm": reconstructor_o,
            "tar_tag": target_class.detach().cpu(),
            "gradient": bdW.detach().cpu(),
            "target_idx": target_idx,
            "base_idx": target_base_idx,
            "base_denorm": base_tensor,
            "psnr": PSNR,
            "ssim": SSIM,
            "lpips": LPIPS,
        })
    else:
        sampler_iter = iter(loader.sampler)
        for _ in range(i*args.batch_size,(i+1)*args.batch_size):
            batch_indices.append(next(sampler_iter))
        print(f"batch_idx {i} batch_indices: {batch_indices}")
        test_tensor = X
        target_labels = y
        test_tensor = torch.clamp(test_tensor.to(ds.device) * ds.view(1,-1,1,1) + dm.view(1,-1,1,1), 0, 1)
        reconstructor_o = torch.clamp(reconstructor_o.to(ds.device) * ds + dm, 0, 1)
        metrix = run_metrics(reconstructor_o, test_tensor, order_batch=True, log=False)

        target_idx = batch_indices[metrix['selector']]
        target_tensor = test_tensor[metrix["selector"]]
        PSNR = metrix['max_psnr']
        SSIM = metrix['max_ssim']
        LPIPS = metrix['max_lpips']
        print(f"target_idx = {target_idx}")
        print(f"PSNR = {PSNR}")
        print(f"SSIM = {SSIM}")
        print(f"LPIPS = {LPIPS}")
        psnrs.append(PSNR)
        ssims.append(SSIM)
        lpips_list.append(LPIPS)
        target_class = target_labels[metrix['selector']]
        TARGET_indices.append(target_idx)
        TARGET_class.append(target_class)
        metrics.append({
            "tar_denorm": target_tensor,
            "rec_denorm": reconstructor_o,
            "tar_tag": target_class.detach().cpu(),
            "gradient": bdW.detach().cpu(),
            "target_idx": target_idx,
            "base_idx": None,
            "base_denorm": None,
            "psnr": PSNR,
            "ssim": SSIM,
            "lpips": LPIPS,
        })
        
Rec_batches = 0
for i in range(len(psnrs)):
    if psnrs[i] >= 19:
        Rec_batches+=1

sorted_psnr = sorted(psnrs,reverse=True)
top_batches_len = int (len(sorted_psnr)*(1/np.e))
PSNR_Top = sorted_psnr[:top_batches_len]
print(f"Rec = {round(Rec_batches/len(psnrs)*100,2)}%")
print(f"PSNR_All: {np.mean(psnrs):.4f} +- {np.std(psnrs):.4f}")
print(f"PSNR_Top: {np.mean(PSNR_Top):.4f} +- {np.std(PSNR_Top):.4f}")
Rec = round(Rec_batches / len(psnrs) * 100, 2)
PSNR_All_mean = np.mean(psnrs)
PSNR_All_std = np.std(psnrs)
PSNR_Top_mean = np.mean(PSNR_Top)
PSNR_Top_std = np.std(PSNR_Top)
print(f"Mean SSIM: {np.mean(ssims):.4f}, Std SSIM: {np.std(ssims):.4f}")
print(f"Mean PSNR: {np.mean(psnrs):.4f}, Std PSNR: {np.std(psnrs):.4f}")
print(f"Mean LPIPS: {np.mean(lpips_list):.4f}, Std LPIPS: {np.std(lpips_list):.4f}")

import pickle

test_results= {
        "metrics": metrics,
        "Rec": Rec,
        "PSNR_All_mean": PSNR_All_mean,
        "PSNR_All_std": PSNR_All_std,
        "PSNR_Top_mean": PSNR_Top_mean,
        "PSNR_Top_std": PSNR_Top_std,
        }

from pathlib import Path

csv_dir = Path(csv_output_dir)
test_dir = Path(test_data_output_dir)

if args.defense in ["LIDS", "LIDS-A"]:
    csv_output_path       = csv_dir / f"psnr_{args.psnr_threshold}.csv"
    item_csv_output_path  = csv_dir / f"itemwise_psnr_{args.psnr_threshold}.csv"
    test_data_output_path = test_dir / f"psnr_threshold-{args.psnr_threshold}.pkl"
else:
    csv_output_path       = csv_dir / f"{args.defense}.csv"
    item_csv_output_path  = csv_dir / f"{args.defense}_itemwise.csv"
    test_data_output_path = test_dir / f"{args.defense}.pkl"   
pickle.dump(test_results, open(test_data_output_path, 'wb'))


with open(csv_output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Mean SSIM", "Std SSIM", 
        "Mean PSNR", "Std PSNR", 
        "Mean LPIPS", "Std LPIPS",
        "Rec", 
        "PSNR_All Mean", "PSNR_All Std",
        "PSNR_Top Mean", "PSNR_Top Std"
    ])
    writer.writerow([
        np.mean(ssims), np.std(ssims), 
        np.mean(psnrs), np.std(psnrs),
        np.mean(lpips_list), np.std(lpips_list),
        Rec, PSNR_All_mean, PSNR_All_std, PSNR_Top_mean, PSNR_Top_std
    ])

print("Summary saved to CSV.")

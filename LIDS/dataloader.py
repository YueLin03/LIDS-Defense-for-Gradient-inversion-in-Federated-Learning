import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from collections import defaultdict
import torchvision.transforms.v2 as v2

class FixDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        if self.transforms:
            data_item = self.transforms(data_item)
        return data_item, self.labels[idx]



class SimpleSampler(torch.utils.data.Sampler):
    def __init__(self, origin_datalen: int, generate_datalen: int,dataset_group_size: int, group_size: int, repeat_num: int, base_to_smote: dict):
        np.random.seed(1337)
        self.origin_datalen = origin_datalen
        self.generate_datalen = generate_datalen
        self.dataset_group_size = dataset_group_size
        self.group_size = group_size
        self.repeat_num = repeat_num
        indices = np.arange(self.origin_datalen)
        expanded_indices = np.tile(indices, 1)
        np.random.shuffle(expanded_indices)
        if (self.repeat_num >= self.group_size-1):
            self.repeat_num = self.group_size-1
        self.final_indices = [[idx for _ in range(self.repeat_num+1)]+[(base_to_smote[idx][i]+self.origin_datalen) for i in range(self.group_size-1 - self.repeat_num) ] for idx in expanded_indices]

        #self.final_indices = [[idx] + [idx * (self.dataset_group_size-1) + i + self.origin_datalen for i in range(self.group_size - 1)] for idx in indices]
        self.flattened_indices = [idx for batch in self.final_indices for idx in batch]
    def __iter__(self):
        return iter(self.flattened_indices)
    
    def __len__(self):
        return self.generate_datalen

class SSampler(torch.utils.data.Sampler):
    def __init__(self, final_indices):
        self.final_indices = final_indices
        self.flattened_indices = [idx for batch in self.final_indices for idx in batch]
        self.gen_len = len(self.flattened_indices)
    def __iter__(self):
        return iter(self.flattened_indices)
    def __len__(self):
        return self.gen_len



def datasets_train_Cifar10(seed=1337):
    torch.manual_seed(seed)
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    data_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
    data_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),

     v2.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_better)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    print(f"trainset len: {len(trainset)}")
    
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # trainset, valset = random_split(trainset, [train_size, val_size])

    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_train_Cifar100(seed=1337):
    torch.manual_seed(seed)
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),

     v2.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_better)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    print(f"trainset len: {len(trainset)}")
    
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # trainset, valset = random_split(trainset, [train_size, val_size])

    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_train_grouped(seed=1337, thresh="18.0", dataset="Cifar100"):
    torch.manual_seed(seed)
    # set to 0.5
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    if dataset == 'Cifar10':
        data_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
        data_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
    elif dataset == 'Cifar100':
        data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
        data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    dataset_path = f"/trainingData/sage/alan/defence_seer/defence/datasets/{dataset}/trainset/init_2000/Norm/MSE/paper/denorm/rand_aug_1/finetune1000epoch_psnr_thresh{thresh}finetune_images_num10000.dst"
    ae_dataset = torch.load(dataset_path, weights_only=False)
    
    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
     v2.Normalize(data_mean, data_std)])

    if dataset == "Cifar100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "Cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    print(f"trainset len: {len(ae_dataset)}")
    sampler = SimpleSampler(
        ae_dataset.origin_datalen,
        len(ae_dataset),
        8,
        int(64 // 16),
        0,
        ae_dataset.base_to_smote
    )
    
    
    train_sampler = SSampler(sampler.final_indices)
    
    ae_dataset = FixDataset(ae_dataset.data, ae_dataset.labels, transforms=transform_better)
    
    trainloader = DataLoader(
        ae_dataset,
        batch_size=64,
        sampler=train_sampler,
        pin_memory=True, num_workers=8,
    )
    
    
    testloader = DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    
    return trainloader, testloader, d_norm, d_denorm



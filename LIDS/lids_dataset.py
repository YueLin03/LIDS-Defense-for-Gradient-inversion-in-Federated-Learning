from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from collections import defaultdict
class LIDS_Dataset(Dataset):
    def __init__(self, data, labels, smote_to_base, smote_to_neighbor, smote_to_alpha, origin_datalen):
        self.data = data
        self.labels = labels
        self.smote_to_base = smote_to_base
        self.smote_to_neighbor = smote_to_neighbor
        self.smote_to_alpha = smote_to_alpha
        self.origin_datalen = origin_datalen
        self.base_to_smote = self._reverse_dict(smote_to_base)
    
    @classmethod
    def _reverse_dict(self, original_dict):
        reversed_dict = defaultdict(list)
        for key, value in original_dict.items():
            reversed_dict[value].append(key)
        return dict(reversed_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleSampler(torch.utils.data.Sampler):
    def __init__(self, origin_datalen: int, generate_datalen: int,dataset_group_size: int, group_size: int, repeat_num: int, base_to_smote: dict):
        np.random.seed(1337)
        self.origin_datalen = origin_datalen
        self.generate_datalen = generate_datalen
        self.dataset_group_size = dataset_group_size
        self.group_size = group_size
        self.repeat_num = repeat_num
        indices = np.arange(self.origin_datalen)
        expanded_indices = np.tile(indices, 4)
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

import torch

# def get_psnr_score(batch_imgs, ref_img, factor=1.0, clip=True):
#     """
#     計算一個 batch 中每張圖片與參考圖片的 PSNR，返回 PSNR 值列表。
    
#     Args:
#         batch_imgs: 一個 batch 的圖片 (Tensor, shape: [batch_size, C, H, W])。
#         ref_img: 參考圖片 (Tensor, shape: [C, H, W])。
#         factor: 最大可能像素值 (默認為 1.0，適用於範圍 [0, 1] 的圖片)。
#         clip: 是否將數據範圍限制在 [0, 1]。

#     Returns:
#         psnr_score_list: 每張圖片的 PSNR 值列表 (list of float)。
#     """
#     if not isinstance(batch_imgs, torch.Tensor):
#         batch_imgs = torch.tensor(batch_imgs, dtype=torch.float32)
#     if not isinstance(ref_img, torch.Tensor):
#         ref_img = torch.tensor(ref_img, dtype=torch.float32)

#     # 確保 batch 和參考圖片形狀正確
#     if len(batch_imgs.shape) != 4:
#         raise ValueError("batch_imgs must have shape [batch_size, C, H, W].")
#     if len(ref_img.shape) != 3:
#         raise ValueError("ref_img must have shape [C, H, W].")
#     if batch_imgs.shape[1:] != ref_img.shape:
#         raise ValueError("All images in batch must have the same shape as ref_img.")

#     if clip:
#         batch_imgs = torch.clamp(batch_imgs, 0, 1)
#         ref_img = torch.clamp(ref_img, 0, 1)

#     psnr_score_list = []

#     for img in batch_imgs:
#         mse = ((img - ref_img) ** 2).mean()
#         if mse > 0 and torch.isfinite(mse):
#             psnr_value = 10 * torch.log10(factor ** 2 / mse)
#             psnr_score_list.append(psnr_value.item())
#         elif mse == 0:
#             psnr_score_list.append(float("inf"))
#         else:
#             psnr_score_list.append(float("nan"))
    
#     return psnr_score_list

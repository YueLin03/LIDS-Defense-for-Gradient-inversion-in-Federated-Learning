import os
import time
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize CIFAR-10 images from normalized tensor format to [0,1] range.
    """
    mean = torch.tensor([0.49146724, 0.48226175, 0.44677013], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.24703224, 0.24348514, 0.26158786], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def get_psnr(img: torch.Tensor, ref: torch.Tensor, factor: float = 1.0, clip: bool = True) -> float:
    """
    Compute PSNR between two images in tensor format (normalized as CIFAR-10).
    """
    # Ensure tensor inputs
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref, dtype=torch.float32)

    # Denormalize for PSNR calculation
    img = denormalize_cifar10(img)
    ref = denormalize_cifar10(ref)

    # Align shapes
    if img.shape != ref.shape:
        raise ValueError("Input images must have the same shape.")

    # Optionally clip to [0,1]
    if clip:
        img = torch.clamp(img, 0.0, 1.0)
        ref = torch.clamp(ref, 0.0, 1.0)

    # Mean Squared Error
    mse = ((img - ref) ** 2).mean()
    if mse == 0:
        return float('inf')
    if not torch.isfinite(mse):
        return float('nan')

    psnr_value = 10 * torch.log10(factor ** 2 / mse)
    return psnr_value.item()


def save_image(image: torch.Tensor, file_path: str) -> None:
    """
    Save a tensor image to disk after denormalizing and plotting.
    """
    # Denormalize and convert to HWC
    img = denormalize_cifar10(image)
    arr = torch.clamp(img, 0.0, 1.0).cpu().numpy()
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.axis('off')
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


class Autoencoder(nn.Module):
    """
    Simple convolutional autoencoder for image encoding/decoding.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 64, 4, 2, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LIDSGroupAutoencoder:
    """
    Generates SMOTE-like samples using latent interpolation with a fine-tuned autoencoder.
    """
    def __init__(self, args, autoencoder: Autoencoder):
        self.args = args
        self.train = args.train == "True"
        self.device = args.device
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # SMOTE settings
        self.psnr_threshold = args.psnr_threshold
        self.init_alpha = args.init_alpha
        self.max_alpha = args.max_alpha
        self.finetune_images_num = args.finetune_images_num
        self.base_num = args.base_num
        self.batch_size = args.batch_size
        self.n_components = args.n_components
        self.class_num = 100 if args.dataset == 'Cifar100' else 10
        self.autoencoder = autoencoder.to(self.device)

        # Logging setup
        log_dir = "./logs/autoencoder"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file = os.path.join(log_dir, f"init_{args.dataset}_{timestamp}.log")
        self.file = open(self.log_file, "w")
        self.file.write("=== Log Start ===\n")

        # Mappings
        self.smote_to_base = {}
        self.base_to_smoteList = {}
        self.smote_to_neighbor = {}
        self.smote_to_alpha = {}
        self.neighbor_list = {}
        self.all_smote_indices = []
    def _prepare_dataset(self, eval_dataset):
        # Extract and normalize dataset vectors
        vectors = np.vstack([X.reshape(-1) for X, y in eval_dataset])
        all_images = np.stack([X for X, y in eval_dataset])
        labels = torch.Tensor([y for _, y in eval_dataset])
        pca = PCA(n_components=self.n_components)
        vectors_transformed = pca.fit_transform(vectors)

        return vectors, vectors_transformed, all_images, labels, pca

    def _get_class_data(self, vectors, all_images, labels, pca):
        # Group data by class for PCA transformation
        xclass_idx = []
        torch_xclass = []
        xclass = []
        yclass = []
        class_vectors_transformed = []

        for c in range(self.class_num):
            class_mask = labels == c
            torch_x_c = all_images[class_mask]
            x_c = vectors[class_mask]
            y_c = labels[class_mask]
            x_idx = np.where(class_mask)[0]

            x_c_transformed = pca.transform(x_c)

            xclass_idx.append(x_idx)
            torch_xclass.append(torch_x_c)
            xclass.append(x_c)
            yclass.append(y_c)
            class_vectors_transformed.append(x_c_transformed)

        return xclass_idx, torch_xclass,xclass, yclass, class_vectors_transformed

    def generate_smote_samples(self, eval_dataset, finetune_epoch=200):
        # Prepare data and apply PCA transformation
        vectors, vectors_transformed, all_images, labels, pca = self._prepare_dataset(eval_dataset)
        all_images_torch = torch.from_numpy(all_images)
        xclass_idx,torch_xclass, xclass, yclass, class_vectors_transformed = self._get_class_data(
            vectors,all_images_torch, labels, pca
        )

        
        smote_img_list = []
        smote_label_list = []

        t0 = time.time()
        sum_of_finetune = 0.0

        Base_table = {}
        finetune_dataidx = set()
        neighbor_table = {}
        autoencoder_table = {}
        neighbor_num = int(self.batch_size // self.base_num)-1
        base_list = []
        # Divide dataset into manageable groups for SMOTE generation
        #total number of autoencoder = n_groups
        # 使用 while 迴圈，確保 eval_dataset 中的每張圖片都被當作 base 處理一次
        current_idx = 0  # 當前處理的圖片索引
        autoencoder_idx = 0 
        finetune_list = [idx for idx in range(len(eval_dataset))]
        autoencoder_finetune_dataidx = {}
        '''
        for xclass in xclass_idx:
            finetune_list.extend(xclass)
        '''
        while current_idx < len(eval_dataset):
            selected_data = []
            # 遍歷 eval_dataset，將 base 和其 neighbors 加入選擇的數據
            while len(finetune_dataidx) <= self.finetune_images_num and current_idx < len(eval_dataset):
                selected_data = []
                base_idx = finetune_list[current_idx]
                
                class_idx = int(labels[base_idx].item())
                base_image = all_images[base_idx]
                base_torch = all_images_torch[base_idx]
                # 計算鄰近點
                real_indices = xclass_idx[class_idx]
                if self.args.img_distance == "MSE":
                    #print(base_torch.shape)
                    #print(torch_xclass[class_idx].shape)
                    with torch.no_grad():
                        distances = torch.nn.functional.mse_loss(base_torch.unsqueeze(0), torch_xclass[class_idx], reduction="none").sum(dim=(1, 2, 3))
                else:
                    distances = np.linalg.norm(
                        vectors_transformed[base_idx] - class_vectors_transformed[class_idx],
                        axis=1
                    )
                neighbor_indices = real_indices[np.argsort(distances)[:neighbor_num + 1]][1:]  # 排除自身
                neighbor_table[base_idx] = neighbor_indices
                self.neighbor_list[base_idx] = neighbor_indices
                '''
                skip = False
                for cur_autoencoder_idx, finetune_dataidx in autoencoder_finetune_dataidx.items():
                    flag =True
                    if (base_idx not in finetune_dataidx):
                        flag = False
                        continue
                    for n_idx in neighbor_table[base_idx]:
                        if n_idx not in finetune_dataidx:
                            flag = False
                            break
                    if(flag == True):
                        Base_table[cur_autoencoder_idx].append(base_idx)
                        skip = True
                        break
                if (skip == True):
                    current_idx += 1
                    continue  
                '''  
                # 添加 base 和 neighbors 到選擇的數據中
                selected_data.append(base_idx)
                selected_data.extend([idx for idx in neighbor_indices])
                base_list.append(base_idx)
                current_idx += 1  # 移動到下一張圖片

                # 將選擇的數據加入 fine-tune 數據庫
                finetune_dataidx.update(selected_data)

            # 如果達到指定的 fine-tune 數量，或者是最後一組，執行 fine-tune
            if len(finetune_dataidx) >= self.args.finetune_images_num or current_idx >= len(eval_dataset)-1:
                

                Base_table[autoencoder_idx] = base_list
                print(f"autoencoder_idx{autoencoder_idx} = {base_list}")
                t_start = time.time()

                # 執行 fine-tune
                finetune_database = [eval_dataset[idx] for idx in finetune_dataidx]
                batch_autoencoder = self._batch_finetune(finetune_database, finetune_epoch, autoencoder_idx)
                t_end = time.time()
                autoencoder_table[autoencoder_idx] = batch_autoencoder

                # 計算 fine-tune 總時間並清空數據庫
                sum_of_finetune += (t_end - t_start)
                autoencoder_finetune_dataidx[autoencoder_idx] = finetune_dataidx
                autoencoder_idx+=1
                finetune_dataidx.clear()  # 清空數據庫以便下一次 fine-tune
                base_list = []

        # Generate SMOTE samples using the trained autoencoders
        for group_idx in range(len(autoencoder_table.items())):
            base_list = Base_table[group_idx]
            batch_autoencoder = autoencoder_table[group_idx] 
            batch_autoencoder.eval()   
            for base_idx in base_list:
                class_idx = int(labels[base_idx].item())
                base_image = all_images[base_idx]
                neighbor_indices = neighbor_table[base_idx]

                smote_img, smote_label, alpha = self._generate_samples(
                    batch_autoencoder,
                    base_image,
                    all_images[neighbor_indices],
                    class_idx
                )    
                smote_indices = list(range(
                    len(self.all_smote_indices),
                    len(self.all_smote_indices) + len(smote_img)
                ))
                self.all_smote_indices.extend(smote_indices)
                self.base_to_smoteList[base_idx] = smote_indices
                for idx in smote_indices:
                    self.smote_to_base[idx] = base_idx
                for n_idx, idx in enumerate(smote_indices):
                    self.smote_to_neighbor[idx] = neighbor_indices[n_idx]
                    self.smote_to_alpha[idx] = alpha[n_idx]
                smote_img_list.extend(smote_img)
                smote_label_list.extend(smote_label)
                if (len(smote_img_list)%100 == 0):
                    print(f"Generate img {len(smote_img_list)}",flush=True)

        smote_dataset = TensorDataset(
            torch.tensor(np.stack(smote_img_list), dtype=torch.float32),
            torch.tensor(smote_label_list, dtype=torch.int64)
        )
        t1 = time.time()
        self.file.write(f"Time of generating image: {t1-t0}\n \
                          Total time for fine-tuning: {sum_of_finetune:.2f} seconds\n")
        print(f"Time of generating image: {t1-t0}")
        print(f"Total time for fine-tuning: {sum_of_finetune:.2f} seconds")
        if self.file:
            self.file.write("=== Log End ===\n")
            self.file.close()
            self.file = None
        alpha_num = {}
        for idx,alpha in self.smote_to_alpha.items():
            alpha = int(round(alpha * 1000,1))
            if alpha not in alpha_num.keys():
                alpha_num[alpha] = 1
            else:
                alpha_num[alpha] += 1
        print("All alpha")
        for alpha, count in sorted(alpha_num.items()):
            print(f"alpha {float(alpha / 1000.0)} = {count}")
        return smote_dataset, self.smote_to_base,self.smote_to_neighbor,self.smote_to_alpha

    def _generate_samples(self, autoencoder, base, neighbors, label):
        """
        Generate interpolated samples using autoencoder with PSNR-based filtering.

        Args:
            autoencoder: Fine-tuned autoencoder
            base: Base image
            neighbors: Neighboring images
            label: Class label

        Returns:
            Generated samples, their labels, and a list of alpha values used.
        """
        base_tensor = torch.tensor(base, dtype=torch.float32, device=self.device).unsqueeze(0)
        neighbor_tensor = torch.tensor(neighbors, dtype=torch.float32, device=self.device)


        # Encode base and neighbor images
        enc_base = autoencoder.encoder(base_tensor)
        enc_neighbors = autoencoder.encoder(neighbor_tensor)

        samples = []
        labels = []
        alphas_used = []

        for i, enc_neighbor in enumerate(enc_neighbors):
            alpha = self.init_alpha
            current_enc = enc_base
            while alpha <= self.max_alpha:
                # Interpolate in latent space
                interpolated_enc = enc_neighbor + alpha * (current_enc - enc_neighbor)

                # Decode the interpolated encoding
                decoded_sample = autoencoder.decoder(interpolated_enc)
                decoded_sample_np = decoded_sample.squeeze(0).detach().cpu().numpy()

                # Calculate PSNR between the decoded sample and base image
                psnr_value = get_psnr(base, decoded_sample_np)

                if psnr_value > self.psnr_threshold:
                    samples.append(decoded_sample_np)
                    labels.append(label)
                    alphas_used.append(alpha)
                    break

                alpha += 0.025
                alpha  = round(alpha, 4)
            # If no valid sample was found, append the final alpha sample
            if alpha > self.max_alpha:
                interpolated_enc = enc_neighbor + self.max_alpha * (current_enc - enc_neighbor)
                decoded_sample = autoencoder.decoder(interpolated_enc)
                decoded_sample_np = decoded_sample.squeeze(0).detach().cpu().numpy()
                samples.append(decoded_sample_np)
                labels.append(label)
                alphas_used.append(self.max_alpha)
            #save_image(decoded_sample_np,f"images/smote/A{i}_psnr{psnr_value}_decoded_img_alpha{alpha}.png")
            #save_image(neighbor_tensor[i],f"images/smote/A{i}_psnr{psnr_value}_neighbor_tensor_alpha{alpha}.png")
            #save_image(base_tensor.squeeze(),f"images/smote/A{i}_psnr{psnr_value}_base_tensor_alpha{alpha}.png")
            
        return np.array(samples), labels, alphas_used

    def _batch_finetune(self, data_batch, epochs: int, batch_idx: int) -> nn.Module:
        """
        Fine-tune the autoencoder on a batch of images.
        """
        # Prepare directories and file paths
        model_dir = f"./weights/{self.args.dataset}/{'trainset' if self.train else 'testset'}/" \
                    f"bs{self.finetune_images_num}_ep{epochs}_base{self.base_num}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"autoencoder_{batch_idx}.pth")

        ae = self.autoencoder
        if os.path.exists(model_path):
            ae.load_state_dict(torch.load(model_path, map_location=self.device))
            return ae

        loader = DataLoader(data_batch, batch_size=128, shuffle=True)
        optimizer = optim.Adam(ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None
        for ep in range(epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                recon = ae(imgs)
                loss = criterion(recon, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = ae.state_dict()

        # Save best model
        torch.save(best_state, model_path)
        ae.load_state_dict(best_state)
        return ae


def train_autoencoder(dataset, output_path: str, log_path: str,
                      epochs: int = 1000, batch_size: int = 16,
                      lr: float = 0.001, device: str = "cpu") -> None:
    """
    Train a standalone autoencoder and save the best model.
    """
    ae = Autoencoder().to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_state = None
    with open(log_path, 'w') as log_file:
        log_file.write("=== Training Log ===\n")
        for ep in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in loader:
                imgs = imgs.to(device)
                recon = ae(imgs)
                loss = criterion(recon, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            log_file.write(f"Epoch {ep+1}/{epochs} Loss: {epoch_loss:.4f}\n")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = ae.state_dict()

    torch.save(best_state, output_path)


def generate_dataset(args, target_dataset):
    """
    Entry point to train or load an autoencoder and produce SMOTE samples.
    """
    # Initialize or load autoencoder
    model_dir = f"./weights/{args.dataset}/{'train' if args.train=='True' else 'test'}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'init{args.init_finetune_epoch}_best_autoencoder.pth')
    log_dir = f"./logs/{args.dataset}/train_{args.train}"
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(model_path):
        train_autoencoder(
            dataset=target_dataset,
            output_path=model_path,
            log_path=os.path.join(log_dir, 'training.log'),
            epochs=args.finetune_epoch,
            batch_size=128,
            lr=0.001,
            device=args.device
        )

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Generate SMOTE-like dataset
    grouper = LIDSGroupAutoencoder(args, autoencoder)
    return grouper.generate_smote_samples(target_dataset, finetune_epoch=args.finetune_epoch)

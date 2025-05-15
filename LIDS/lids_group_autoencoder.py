import numpy as np
from sklearn.neighbors import NearestNeighbors
import collections
import time
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
import matplotlib.pyplot as plt

# OS
import os
import argparse

from PIL import Image
def denormalize_cifar10(tensor):
    """
    Denormalize CIFAR10 images from tensor format to [0,1] range
    """
    mean = torch.tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324]).view(3, 1, 1).to(tensor.device)
    
    return tensor * std + mean
def get_psnr(img, ref, factor=1.0, clip=True):
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref, dtype=torch.float32)
    img = denormalize_cifar10(img)
    ref = denormalize_cifar10(ref)
    if img.shape != ref.shape:
        raise ValueError("Input images must have the same shape.")

    if clip:
        img = torch.clamp(img, 0, 1)
        ref = torch.clamp(ref, 0, 1)
    mse = ((img - ref) ** 2).mean()

    if mse > 0 and torch.isfinite(mse):
        psnr_value = 10 * torch.log10(factor ** 2 / mse)
        return psnr_value.item()  
    elif mse == 0:
        return float("inf")
    else:
        return float("nan") 


def save_image(image, file_path):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    image = denormalize_cifar10(image)
    image_array = torch.clamp(image, 0, 1).cpu().numpy()
    if image_array.shape[0] == 3: 
        image_array = np.transpose(image_array, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    ax.axis('off')  # Remove axes for a clean image
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to avoid memory issues

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=4, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LIDSGroupAutoencoder:
    def __init__(self, args, autoencoder):
        self.args = args
        if args.train == "True":
            self.train = True
        elif args.train == "False":
            self.train = False
        self.device = args.device
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.psnr_threshold = args.psnr_threshold
        self.init_alpha = args.init_alpha
        self.max_alpha = args.max_alpha
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.autoencoder = autoencoder
        self.finetune_images_num = args.finetune_images_num
        self.n_components = args.n_components
        self.base_num = args.base_num
        self.class_num = 100 if args.dataset == 'Cifar100' else 10
        self.dataset = args.dataset 
        self.all_smote_indices = []
        self.all_base_indices = []
        self.groups = []
        self.neighbor_list = {}
        self.base_to_smoteList = {}
        self.smote_to_base = {}
        self.smote_to_neighbor = {}
        self.smote_to_alpha = {}
        self.img_distance = args.img_distance
        log_dir = "./logs/autoencoder"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True) 
        self.log_file = log_dir + f'/init_Train_{self.train}_dataset_{self.dataset}_base_{self.base_num}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'

        self.file = open(self.log_file, "w")
        self.file.write("=== Log Start ===\n")

    def _prepare_dataset(self, target_dataset):
        # Extract and normalize dataset vectors
        vectors = np.vstack([X.reshape(-1) for X, y in target_dataset])
        all_images = np.stack([X for X, y in target_dataset])
        labels = torch.Tensor([y for _, y in target_dataset])
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

    def generate_smote_samples(self, target_dataset, finetune_epoch=200):
        """
        1) Randomly split dataset into groups of size self.finetune_images_num
        2) Fine-tune one AE per group
        3) For each base image, find k-nearest neighbors among same-class images in its group
        4) Generate SMOTE samples via latent interpolation
        """
        # Prepare flattened vectors, PCA, arrays and labels
        vectors, vectors_pca, images_np, labels, pca = self._prepare_dataset(target_dataset)
        images_t = torch.from_numpy(images_np).to(self.device)
        
        # how many neighbors per base
        k = max(1, self.batch_size // self.base_num - 1)
        
        # random non-overlapping groups
        idxs = np.arange(len(target_dataset))
        np.random.shuffle(idxs)
        groups = [idxs[i : i + self.finetune_images_num]
                  for i in range(0, len(idxs), self.finetune_images_num)]
        
        smote_imgs, smote_lbls = [], []
        t0 = time.time()
        total_ft = 0.0
        
        for gid, grp in enumerate(groups):
            # fine-tune AE on this group
            group_data = [target_dataset[i] for i in grp]
            t_start = time.time()
            ae = self._batch_finetune(group_data, finetune_epoch, gid)
            total_ft += time.time() - t_start
            ae.eval()
            
            # map class → indices within this group
            grp_labels = labels[grp].cpu().numpy()
            class_map = {}
            for idx, lbl in zip(grp, grp_labels):
                class_map.setdefault(int(lbl), []).append(idx)
            
            # generate samples per base
            for base_idx in grp:
                lbl = int(labels[base_idx].item())
                same_cls = class_map.get(lbl, [])
                if len(same_cls) < 2:
                    continue  # no neighbors
                
                # compute distances among same-class
                if self.img_distance == "MSE":
                    d = F.mse_loss(
                        images_t[base_idx].unsqueeze(0),
                        images_t[same_cls],
                        reduction="none"
                    ).view(len(same_cls), -1).sum(dim=1).cpu().numpy()
                else:
                    emb = vectors_pca
                    d = np.linalg.norm(emb[base_idx] - emb[same_cls], axis=1)
                
                # pick k nearest excluding self
                nbrs_idx = np.argsort(d)
                nbrs_idx = [same_cls[i] for i in nbrs_idx if i != same_cls.index(base_idx)]
                nbrs_idx = nbrs_idx[:k]
                neighbors = images_np[nbrs_idx]
                
                # generate and record SMOTE samples
                samples, lbls_out, alphas = self._generate_samples(
                    ae, images_np[base_idx], neighbors, lbl
                )
                for s, l, a in zip(samples, lbls_out, alphas):
                    idx = len(smote_imgs)
                    smote_imgs.append(s)
                    smote_lbls.append(l)
                    self.smote_to_base[idx] = base_idx
                    self.smote_to_neighbor[idx] = nbrs_idx[len(self.smote_to_neighbor) % k]
                    self.smote_to_alpha[idx] = a
                
                if len(smote_imgs) % 100 == 0:
                    print(f"Generated {len(smote_imgs)} samples")
        
        ds = TensorDataset(
            torch.tensor(np.stack(smote_imgs), dtype=torch.float32),
            torch.tensor(smote_lbls, dtype=torch.int64)
        )
        print(f"Fine-tuning: {total_ft:.1f}s, total elapsed: {time.time()-t0:.1f}s")
        return ds, self.smote_to_base, self.smote_to_neighbor, self.smote_to_alpha


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
    
    def _batch_finetune(self, selected_data,finetune_epoch,Batch_index):
        """
        Fine-tune autoencoder on a selected batch of data.
        
        Args:
            selected_data: Batch of data for fine-tuning
        
        Returns:
            Fine-tuned autoencoder
        """
        # Implement fine-tuning logic here
        # This is a placeholder - you'd need to add actual fine-tuning implementation
        batch_autoencoder  = self.autoencoder.to(self.device)
        num_epochs = finetune_epoch
        if self.img_distance == "MSE":
            if self.train == True:
                model_dir = f"./weights/{self.dataset}/trainset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{num_epochs}/base_num{self.base_num}/mse"
            else:
                model_dir = f"./weights/{self.dataset}/testset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{num_epochs}/base_num{self.base_num}/mse"
        else:        
            if self.train == True:
                model_dir = f"./weights/{self.dataset}/trainset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{num_epochs}/base_num{self.base_num}"
            else:
                model_dir = f"./weights/{self.dataset}/testset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{num_epochs}/base_num{self.base_num}"   
        model_path = model_dir+ f"/init_autoencoder_{Batch_index}.pkl"
        if os.path.exists(model_path):
            self.file.write(f"Batch {Batch_index}Load cached fintune autoencoder\n")
            print("Load cached fintune autoencoder")
            ckpt = torch.load(model_path, map_location="cpu")
            batch_autoencoder.load_state_dict(ckpt)
            batch_autoencoder = batch_autoencoder.to(self.device)
        else:
            self.file.write(f'Autoencoder Training seed {self.seed},base_num{self.base_num}, Batch_index {Batch_index}, len of the batch {len(selected_data)}\n')
            print(f'Autoencoder Training seed {self.seed} Batch_index {Batch_index} len of the batch {len(selected_data)}')
            # Define an optimizer and criterion
            criterion = nn.MSELoss()
            optimizer = optim.Adam(batch_autoencoder.parameters(), lr=0.001)

            trainloader = torch.utils.data.DataLoader(selected_data, batch_size=128,
                                                        shuffle=True, num_workers=2)
            
            best_loss  = float('inf')
            best_loss_epoch = -1 
            best_model_state = None 
            for epoch in range(num_epochs):
                for data in trainloader:
                    img, _ = data
                    img = img.to(self.device)
                    
                    output = batch_autoencoder(img)
                    loss = criterion(output, img)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if ((epoch+1)%50 == 0):
                    self.file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n')
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                if loss.item() < best_loss:
                    best_loss_epoch = epoch+1
                    best_loss = loss.item()
                    best_model_state = batch_autoencoder.state_dict()
                if ((epoch+1)%100 == 0):
                    saved_epoch = epoch+1
                    print(f"Saved checkpoint, Epoch {saved_epoch}")
                    print(f'bst epoch: {best_loss_epoch}, bst Loss: {best_loss:.4f}')
                    
                    if self.train == True:
                        checkpoint_dir =  f"./weights/{self.dataset}/trainset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{saved_epoch}/base_num{self.base_num}"
                    else:
                        checkpoint_dir =  f"./weights/{self.dataset}/testset/{self.seed}/finetune_images-{self.finetune_images_num}/num_epochs_{saved_epoch}/base_num{self.base_num}"
                    if self.img_distance == "MSE":    
                        checkpoint_dir = checkpoint_dir +"/mse"
                    checkpoint_path = checkpoint_dir + f"/init_autoencoder_{Batch_index}.pkl"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir, exist_ok=True) 
                    torch.save(best_model_state, checkpoint_path)                     
            self.file.write(f'best_loss_epoch:{best_loss_epoch} bst Loss: {best_loss:.4f}\n')
            print(f'bst Loss: {best_loss:.4f}')
            batch_autoencoder.load_state_dict(best_model_state)
            self.file.write('Finished Training\n')
            self.file.write(f'Saving Model of seed_{self.seed}_{Batch_index}_{num_epochs}\n')
            print('Finished Training')
            print(f'Saving Model of seed_{self.seed}_{Batch_index}_{num_epochs}')
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True) 
            torch.save(batch_autoencoder.state_dict(), model_path)            
        return batch_autoencoder
def train_autoencoder(target_dataset, output_path, log_path, num_epochs=1000, batch_size=16, lr=0.001, device="cpu"):
    autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    trainloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    best_loss = float('inf')
    best_model_state = None

    with open(log_path, 'w') as log_file:
        log_file.write("=== Training Log ===\n")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for data in trainloader:
                img, _ = data
                img = img.to(device)

                output = autoencoder(img)
                loss = criterion(output, img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(trainloader)

            log_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}\n')
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            if epoch_loss < best_loss:
                bst_epoch = epoch
                best_loss = epoch_loss
                best_model_state = autoencoder.state_dict()

        torch.save(best_model_state, output_path)
        log_file.write(f"Training completed. Bestepoch: {bst_epoch}. Best Loss: {best_loss:.4f}\n")
        log_file.write("=== Training Log End ===\n")

    print(f"Training completed. Bestepoch: {bst_epoch} Best Loss: {best_loss:.4f}")


def generate_dataset(args,target_dataset):
    """
    Wrapper function to maintain original interface while using new class.
    """
    print(f"finetune_epoch = {args.finetune_epoch},seed = {args.seed},")
    # Load pre-trained autoencoder
    autoencoder = Autoencoder()
    if args.train == "True":
        output_dir = f"./weights/{args.dataset}/train/"
        output_path = f"./weights/{args.dataset}/train/init_bst_autoencoder_bs1024.pth"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True) 
    else:
        output_dir = f"./weights/{args.dataset}/test/"
        output_path = f"./weights/{args.dataset}/test/init_bst_autoencoder200_bs1024.pth"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True) 
    log_dir=f"./logs/{args.dataset}/train_{args.train}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    #output_path = "./weights/Cifar10/autoencoder_cifar10.pth"
    if not os.path.exists(output_path):
        train_autoencoder(
            target_dataset=target_dataset,
            output_path=output_path,
            log_path=f"./logs/{args.dataset}/train_{args.train}/init_autoencoder2000_bs1024_training_log.txt",
            num_epochs=2000,
            batch_size=128,
            lr=0.001,
            device=args.device
        )  
    autoencoder.load_state_dict(torch.load(output_path, map_location="cpu"))
  
    # Initialize and run grouping
    grouper = LIDSGroupAutoencoder(
        args, 
        autoencoder
    )
    
    # Generate SMOTE samples
    smote_dataset, smote_to_base,smote_to_neighbor,smote_to_alpha = grouper.generate_smote_samples(
        target_dataset, 
        finetune_epoch = args.finetune_epoch
    )
    
    
    return (
        smote_dataset, 
        smote_to_base,
        smote_to_neighbor,
        smote_to_alpha
    )



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
        self.smote_to_neighbor = {}
        self.smote_to_alpha = {}

    def _prepare_dataset(self, target_dataset):
        """
        Flatten images and apply PCA for latent interpolation.
        """
        images = np.stack([x for x, _ in target_dataset])
        vectors = images.reshape(len(images), -1)
        labels = np.array([y for _, y in target_dataset])
        pca = PCA(n_components=self.n_components)
        vectors_pca = pca.fit_transform(vectors)
        return images, vectors_pca, labels, pca

    def generate_smote_samples(self, target_dataset, finetune_epoch: int = 200):
        """
        Create synthetic samples by interpolating latent representations.
        """
        images_np, vectors_pca, labels, pca = self._prepare_dataset(target_dataset)
        images_t = torch.from_numpy(images_np).to(self.device)

        # Define group splits
        idxs = np.arange(len(target_dataset))
        np.random.shuffle(idxs)
        groups = np.array_split(idxs, len(idxs) // self.finetune_images_num)

        samples, sample_labels, alphas = [], [], []
        total_time = 0
        for gid, grp in enumerate(groups):
            start = time.time()
            ae = self._batch_finetune([target_dataset[i] for i in grp], finetune_epoch, gid)
            total_time += time.time() - start

            # Class-wise neighbor selection
            for base_idx in grp:
                lbl = int(labels[base_idx])
                same_cls = grp[labels[grp] == lbl]
                if len(same_cls) < 2:
                    continue

                # Compute distances
                if self.args.img_distance == "MSE":
                    dists = F.mse_loss(
                        images_t[base_idx].unsqueeze(0),
                        images_t[same_cls],
                        reduction="none"
                    ).view(len(same_cls), -1).sum(dim=1).cpu().numpy()
                else:
                    dists = np.linalg.norm(vectors_pca[base_idx] - vectors_pca[same_cls], axis=1)

                # Select nearest neighbors
                nbrs = [same_cls[i] for i in np.argsort(dists)[: max(1, len(grp)//self.base_num - 1)]]

                # Generate samples per neighbor
                for nbr in nbrs:
                    out_samps, out_lbls, out_als = self._generate_samples(
                        ae, images_np[base_idx], images_np[nbr], lbl
                    )
                    samples.extend(out_samps)
                    sample_labels.extend(out_lbls)
                    alphas.extend(out_als)

        dataset = TensorDataset(
            torch.tensor(np.stack(samples), dtype=torch.float32),
            torch.tensor(sample_labels, dtype=torch.long)
        )
        print(f"Generated {len(samples)} latent samples in {total_time:.1f}s")
        return dataset, self.smote_to_base, self.smote_to_neighbor, self.smote_to_alpha

    def _generate_samples(self, ae, base, neighbor, label):
        """
        Interpolate between base and neighbor in latent space and filter by PSNR.
        """
        base_t = torch.tensor(base, dtype=torch.float32, device=self.device).unsqueeze(0)
        nbr_t = torch.tensor(neighbor, dtype=torch.float32, device=self.device).unsqueeze(0)

        enc_base = ae.encoder(base_t)
        enc_nbr = ae.encoder(nbr_t)

        out_samps, out_labels, out_als = [], [], []
        alpha = self.init_alpha
        while alpha <= self.max_alpha:
            interp = enc_nbr + alpha * (enc_base - enc_nbr)
            dec = ae.decoder(interp).squeeze(0).detach().cpu().numpy()
            if get_psnr(base_t.squeeze(), dec) > self.psnr_threshold:
                out_samps.append(dec)
                out_labels.append(label)
                out_als.append(alpha)
                break
            alpha = round(alpha + 0.025, 4)

        # If no sample passes threshold, use max_alpha
        if alpha > self.max_alpha:
            interp = enc_nbr + self.max_alpha * (enc_base - enc_nbr)
            dec = ae.decoder(interp).squeeze(0).detach().cpu().numpy()
            out_samps.append(dec)
            out_labels.append(label)
            out_als.append(self.max_alpha)

        return out_samps, out_labels, out_als

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
    model_path = os.path.join(model_dir, 'best_autoencoder.pth')
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

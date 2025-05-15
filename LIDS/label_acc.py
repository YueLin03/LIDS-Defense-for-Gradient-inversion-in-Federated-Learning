import argparse
import pickle
import torch
import torchvision
from model import Classifier
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='label acc')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--attack_prop', type=str, default="bright")
# parser.add_argument('--loader_type', type=str, default="random")
parser.add_argument('--thresh', type=str, default="15.0")
# parser.add_argument('--repeat', type=str, default="0")
parser.add_argument('--dataset', type=str, default="Cifar100")
parser.add_argument('--basenum', type=int, default=16)


args = parser.parse_args()
attack_prop = args.attack_prop
thresh = args.thresh
dataset = args.dataset
device = args.device
base_num = args.basenum
print(str(args))

resize_transform = torchvision.transforms.Resize(224)

class ImageTagDataset(Dataset):
    def __init__(self, rec_img, tar_tags):
        self.rec_img = rec_img
        self.tar_tags = tar_tags

    def __len__(self):
        return self.rec_img.shape[0]

    def __getitem__(self, idx):
        return self.rec_img[idx], self.tar_tags[idx]

# TODO: change base path
base_path = f"/test_data/paper/{dataset}/OGM/Norm/{attack_prop}/base_num_{base_num}/repeat_num_0/psnr_threshold-{thresh}.pkl"
results: list[dict] = pickle.load(open(base_path, 'rb'))

if dataset == "Cifar10":
    model = Classifier(num_classes=10)
    ckpt = torch.load(f'weights/dinov2_CIFAR10.pt', weights_only=False)

elif dataset == "Cifar100":
    model = Classifier(num_classes=100)
    ckpt = torch.load(f'weights/dinov2_CIFAR100.pt', weights_only=False)

model.load_state_dict(ckpt)
model = model.to(device)

rec_img = []
tar_img = []
tar_tags = []

for i, result in enumerate(results['metrics']):
    rec_tensor: torch.Tensor = result['rec_denorm'].detach()
    # tar_tensor: torch.Tensor = result['tar_denorm'].detach()
    tar_tag = result['tar_tag']
    rec_img.append(rec_tensor.squeeze())
    # tar_img.append(tar_tensor.squeeze())
    tar_tags.append(tar_tag)


rec_img = torch.stack(rec_img)
# tar_img = torch.stack(tar_img)
tar_tags = torch.stack(tar_tags).squeeze()
# print(rec_img.shape)
# print(tar_tags.shape)
assert rec_img.size(0) == tar_tags.size(0), "size mismatch"

datasets = ImageTagDataset(rec_img, tar_tags)
dataloader = DataLoader(datasets, batch_size=32, num_workers=4)

total = 0
correct = 0

for inputs, labels in dataloader:
    inputs = resize_transform(inputs)
    inputs, labels = inputs.to(device), labels.to(device)
    # print(inputs.shape, labels.shape)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # Get the class with the highest probability
    
    # print(preds.shape)
    # print((preds == labels).shape)
    # print(preds == labels)
    total += labels.size(0)
    correct += (preds == labels).sum().item()

print(f'correct: {correct}, total: {total} acc: {correct / total * 100:.4f}%')

with open(f'logs/RLA_results.txt', 'a+', encoding='utf-8') as f:
    f.write(f"prop {attack_prop}, thresh {thresh}, {dataset}, {correct / total * 100:.4f}\n")

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
import os
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
from model import ResNet18
from dataloader import *
import time
import numpy as np

torch.set_float32_matmul_precision('high')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='CIFAR Training with ResNet18')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='Cifar100')
parser.add_argument('--lids', action='store_true', help='Use grouped dataset')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=12)
parser.add_argument('--device', type=str, default="cuda:1")
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--thresh', type=str, default="15.0")

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.num_epochs
device = args.device
weight_decay = args.weight_decay
thresh = args.thresh
dataset = args.dataset
img_distance = "MSE"
augment = "Norm"
finetune_epoch = 1000
iter_per_test = 200

print(args)

if args.lids:
    trainloader, testloader, d_norm, d_denorm = datasets_train_grouped(seed=SEED, thresh=thresh, dataset=dataset)
    if dataset == 'Cifar100':
        num_classes = 100
    elif dataset == 'Cifar10':
        num_classes = 10
else:
    if dataset == 'Cifar100':
        trainset, testset, d_norm, d_denorm = datasets_train_Cifar100(seed=SEED)
        num_classes = 100
    elif dataset == 'Cifar10':
        trainset, testset, d_norm, d_denorm = datasets_train_Cifar10(seed=SEED)
        num_classes = 10

    print(f"trainset len: {len(trainset)}")
    print(f"testset len: {len(testset)}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

model = ResNet18(seed=SEED, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'bn' in n], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'bn' not in n], 'weight_decay': weight_decay}
], lr=learning_rate)

scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=len(trainloader))

model = torch.compile(model)

def train(epoch, logfile):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):        
        # if inputs.shape[0] != 64:
        #     continue
        
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

            
    model.eval()
    correct = 0
    total = 0
    saw_batch = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in testloader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            saw_batch += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    logfile.write(f'Epoch [{epoch+1}/{num_epochs}], Test loss: {running_loss / saw_batch:.4f}, Test Accuracy: {100 * correct / total:.2f}%\n')
    print(f'Test loss: {running_loss / saw_batch:.4f}, Test Accuracy: {100 * correct / total:.2f}%')
    

            
def test(logfile):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    logfile.write(f'Test Accuracy: {100 * correct / total:.2f}%\n')
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    return correct / total
    

if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lids = '_grouped' if args.lids else ''
    logfile = open(f'logs/resnet18_{args.dataset}{lids}.log', 'w', encoding="utf-8")
    logfile.write(str(args)+'\n')
    start = time.time()
    for epoch in range(num_epochs):
        train(epoch, logfile)
    test(logfile)
    print('Finished Training')
    end = time.time()
    print(f'Total time: {end - start:2f}s')
    logfile.write(f"\ntotal time: {end - start:.2f}s\n")

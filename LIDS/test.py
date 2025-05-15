from model import Classifier
import torch

model = Classifier(num_classes=10)
ckpt = torch.load(f'/trainingData/sage/alan/dinov2-finetune/weights/dinov2_CIFAR10.pt', weights_only=False)
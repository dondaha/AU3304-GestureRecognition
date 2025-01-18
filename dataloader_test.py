import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from tqdm import tqdm

import torch
import numpy as np
import random
import os

def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False
    

setup_seed(1231312)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    # 正则化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='data/rps', transform=transform)
test_dataset = datasets.ImageFolder(root='data/rps-test-set', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset information
print(f'Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}')
print(f'Classes: {train_dataset.classes}')

# Print the number of samples per class in the training dataset
class_counts = {class_name: 0 for class_name in train_dataset.classes}
for _, label in train_dataset.samples:
    class_counts[train_dataset.classes[label]] += 1
print(f'Training samples per class: {class_counts}')

# Print the number of samples per class in the testing dataset
class_counts = {class_name: 0 for class_name in test_dataset.classes}
for _, label in test_dataset.samples:
    class_counts[test_dataset.classes[label]] += 1
print(f'Testing samples per class: {class_counts}')
def sample_images_from_loader(loader, num_samples=20):
    sampled_indices = random.sample(range(len(loader.dataset.samples)), num_samples)
    for idx in sampled_indices:
        image_path, label = loader.dataset.samples[idx]
        print(f'Image Path: {image_path}, Label: {loader.dataset.classes[label]}')

# Sample images from train and test loaders
print('Sampled training images from DataLoader:')
sample_images_from_loader(train_loader)

print('Sampled testing images from DataLoader:')
sample_images_from_loader(test_loader)

# data_analyse.py 2025.1.18 by ddh
# 数据分析脚本
# 从data/目录下读取数据，分析数据的分布情况，并可视化展示。

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 创建输出目录
output_dir = 'output/data_analyse'
os.makedirs(output_dir, exist_ok=True)

# 数据transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees=15, fill=255),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.1])
])

# 构建Dataloader
batch_size = 128
train_dataset = datasets.ImageFolder(root='data/rps', transform=transform)
test_dataset = datasets.ImageFolder(root='data/rps-test-set', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 数据集信息
train_class_counts = {class_name: 0 for class_name in train_dataset.classes}
for _, label in train_dataset.samples:
    train_class_counts[train_dataset.classes[label]] += 1

test_class_counts = {class_name: 0 for class_name in test_dataset.classes}
for _, label in test_dataset.samples:
    test_class_counts[test_dataset.classes[label]] += 1

# 打印数据集信息
print(f'Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}')
print(f'Classes: {train_dataset.classes}')

# 可视化类别分布
def plot_class_distribution(class_counts, dataset_type):
    plt.figure(figsize=(12, 8), dpi=150)
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title(f'{dataset_type} Set Class Distribution', fontsize=25)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_type}_class_distribution.png'))
    plt.close()

plot_class_distribution(train_class_counts, 'Training')
plot_class_distribution(test_class_counts, 'Testing')

print(f'Visualizations saved to {output_dir}')
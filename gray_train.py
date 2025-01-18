from cv2 import log
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from thop import profile  # Import thop for calculating FLOPs and parameters


# 配置logger，记录存储到{output_dir}目录下，以当前时间命名
from datetime import datetime
from logging import FileHandler
from logging import Formatter
from logging import StreamHandler
from logging import getLogger

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = getLogger()
    logger.setLevel('INFO')
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = FileHandler(f'{output_dir}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

# 设置输出目录
i = 1
while os.path.exists(os.path.abspath(f'output/train_{i}')):
    i += 1
output_dir = os.path.abspath(f'output/train_{i}')
logger = setup_logger(output_dir)

# 配置随机种子，保证结果可复现
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
    logger.info(f'Setup seed: {seed}')
    
setup_seed(1231312)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

# 数据transform
transform = transforms.Compose([
    # 转换为灰度图，num_output_channels=1表示单通道
    transforms.Grayscale(num_output_channels=1),
    # 随机旋转角度
    transforms.RandomRotation(degrees=15, fill=255),
    # 缩放到100*100
    transforms.Resize((100, 100)),
    # 转换为Tensor
    transforms.ToTensor(),
    # 归一化到 [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])
])

## 测试transform，并输出可视化图片
os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
visualize_dir = os.path.join(output_dir, "visualization")
## 测试data/rps/paper/paper01-000.png经过transform后的效果
source_img_path = 'data/rps/paper/paper01-000.png'
### 复制原图到可视化目录
shutil.copy(source_img_path, visualize_dir)
img = Image.open(source_img_path)
img = transform(img)
img = img.squeeze().numpy()
img = (img + 1) / 2
plt.imsave(os.path.join(visualize_dir, 'transformation.png'), img, cmap='gray')
logger.info(f'Visualization of transformation saved to {os.path.join(visualize_dir, "transformation.png")}')

# 构建Dataloader
batch_size = 32
train_dataset = datasets.ImageFolder(root='data/rps', transform=transform)
test_dataset = datasets.ImageFolder(root='data/rps-test-set', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## 输出dataset information
logger.info(f'Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}')
logger.info(f'Classes: {train_dataset.classes}')

## Print the number of samples per class in the training set
train_class_counts = {class_name: 0 for class_name in train_dataset.classes}
for _, label in train_dataset.samples:
    train_class_counts[train_dataset.classes[label]] += 1

logger.info("Training set class distribution:")
for class_name, count in train_class_counts.items():
    logger.info(f'{class_name}: {count}')

## Print the number of samples per class in the testing set
test_class_counts = {class_name: 0 for class_name in test_dataset.classes}
for _, label in test_dataset.samples:
    test_class_counts[test_dataset.classes[label]] += 1

logger.info("Testing set class distribution:")
for class_name, count in test_class_counts.items():
    logger.info(f'{class_name}: {count}')
    
# 构建 model, loss function, and optimizer
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
test_interval = 1  # Test the model every epoch

def get_current_lr(epoch, optimizer):
    """定义学习率策略，获取当前学习率"""
    lr = 0.001
    if epoch >= 10:
        lr = lr / 100
    elif epoch >= 5:
        lr = lr / 10
    logger.info(f'Current learning rate: {lr}')
    return lr

# Initialize lists to store learning rate, loss, and accuracy for each epoch
learning_rates = []
epoch_losses = []
epoch_accuracies = []
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Update learning rate
    current_lr = get_current_lr(epoch, optimizer)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    learning_rates.append(current_lr)

    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')
    
    # Test the model at specified intervals
    if (epoch + 1) % test_interval == 0:
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        misclassified_images = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect all labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # Collect misclassified images
                for i in range(len(labels)):
                    if predicted[i] != labels[i]:
                        misclassified_images.append((test_loader.dataset.samples[i][0], predicted[i].item(), labels[i].item()))
        
        accuracy = 100 * correct / total
        epoch_accuracies.append(accuracy)
        logger.info(f'Accuracy after epoch {epoch+1}: {accuracy}%')
        
        cm = confusion_matrix(all_labels, all_predictions)
        logger.info("Confusion Matrix:")
        logger.info("\n"+str(cm))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'Best model saved with accuracy: {best_accuracy}%')

torch.save(model.state_dict(), os.path.join(output_dir, 'last_model.pth'))
logger.info('Last model saved')

# Calculate model parameters and FLOPs
dummy_input = torch.randn(1, 1, 100, 100).to(device)
flops, params = profile(model, inputs=(dummy_input,))
logger.info(f'Model parameters: {params}')
logger.info(f'Model FLOPs: {flops}')

# Plot learning rate, loss, and accuracy
plt.figure(dpi=150)
plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate per Epoch')
plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to be integers
plt.legend()
plt.savefig(os.path.join(visualize_dir, 'learning_rate.png'))
logger.info(f'Learning rate plot saved to {os.path.join(visualize_dir, "learning_rate.png")}')

plt.figure(dpi=150)
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to be integers
plt.legend()
plt.savefig(os.path.join(visualize_dir, 'loss.png'))
logger.info(f'Loss plot saved to {os.path.join(visualize_dir, "loss.png")}')

plt.figure(dpi=150)
plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to be integers
plt.legend()
plt.savefig(os.path.join(visualize_dir, 'accuracy.png'))
logger.info(f'Accuracy plot saved to {os.path.join(visualize_dir, "accuracy.png")}')
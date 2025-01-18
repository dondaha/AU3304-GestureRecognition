import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
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

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

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

# Print the number of samples per class in the training set
train_class_counts = {class_name: 0 for class_name in train_dataset.classes}
for _, label in train_dataset.samples:
    train_class_counts[train_dataset.classes[label]] += 1

print("Training set class distribution:")
for class_name, count in train_class_counts.items():
    print(f'{class_name}: {count}')

# Print the number of samples per class in the testing set
test_class_counts = {class_name: 0 for class_name in test_dataset.classes}
for _, label in test_dataset.samples:
    test_class_counts[test_dataset.classes[label]] += 1

print("Testing set class distribution:")
for class_name, count in test_class_counts.items():
    print(f'{class_name}: {count}')

# Initialize model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
test_interval = 1  # Test the model every epoch

for epoch in range(num_epochs):
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
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    
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
        print(f'Accuracy after epoch {epoch+1}: {accuracy}%')
        
        # Print misclassified images
        # if misclassified_images:
        #     print("Misclassified images:")
        #     for img_path, pred_label, true_label in misclassified_images:
        #         print(f'Image: {img_path}, Predicted: {pred_label}, True: {true_label}')
        
        # Compute and print confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:")
        print(cm)
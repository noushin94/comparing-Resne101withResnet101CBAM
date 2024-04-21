import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet101, ResNet101_Weights

# Import the resnet_cbam module which you provided
from resnet_cbam import resnet101_cbam

# Configuration
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_resnet101 = resnet101(weights=ResNet101_Weights.DEFAULT).to(device)
model_resnet101_cbam = resnet101_cbam(pretrained=False).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet101 = optim.Adam(model_resnet101.parameters(), lr=learning_rate)
optimizer_resnet101_cbam = optim.Adam(model_resnet101_cbam.parameters(), lr=learning_rate)

# Training function
def train_model(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training and evaluating
for epoch in range(num_epochs):
    loss_resnet101 = train_model(model_resnet101, optimizer_resnet101, train_loader)
    loss_resnet101_cbam = train_model(model_resnet101_cbam, optimizer_resnet101_cbam, train_loader)
    
    acc_resnet101 = evaluate_model(model_resnet101, test_loader)
    acc_resnet101_cbam = evaluate_model(model_resnet101_cbam, test_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], ResNet101 Loss: {loss_resnet101:.4f}, Accuracy: {acc_resnet101:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], ResNet101 CBAM Loss: {loss_resnet101_cbam:.4f}, Accuracy: {acc_resnet101_cbam:.2f}%')

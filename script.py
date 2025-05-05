import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

# Device configuration (Use GPU if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Define transformations for the CIFAR-10 dataset
transform_train = transforms.Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Define the VGG16-like model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv6
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 3
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Conv8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv9
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pool 4
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv11
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv12
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pool 5
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # Adjust input size according to CIFAR-10's image size (32x32)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)  # Output layer (CIFAR-10 has 10 classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc_layers(x)
        return x


data_path = './data/cifar-10-batches-py'

if os.path.exists(data_path):
    print("✅ CIFAR-10 ya está descargado.")
    download_flag = False
else:
    print("⬇️ Descargando CIFAR-10...")
    download_flag = True

train_dataset = datasets.CIFAR10(root='./data', train=True, download=download_flag, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=download_flag, transform=transform_test)
# Data loaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = VGG16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

import time

# Training the model
num_epochs = 64
total_start = time.time()  # Tiempo inicial del entrenamiento

train_losses = []
test_accuracies = []
checkpoints = [10, 20, 32, 48, 64]
best_accuracy = 0.0

for epoch in range(num_epochs):
    start_time = time.time()  # Tiempo inicial por época

    model.train()  # Set the model to training mode
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluación por época
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    scheduler.step(accuracy)

    # Guardar el mejor modelo
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_vgg16_model.pth')

    # Guardar checkpoints clave
    if (epoch + 1) in checkpoints:
        torch.save(model.state_dict(), f'vgg16_epoch_{epoch+1}.pth')
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_duration:.2f}s")

total_time = time.time() - total_start
print(f"\n⏱️ Entrenamiento completo en {total_time:.2f} segundos ({total_time/60:.2f} min)")

# Testing the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10,000 test images: {100 * correct / total:.2f}%')


# Guardar el modelo entrenado
#torch.save(model.state_dict(), 'vgg16_cifar10_mps.pth')
#print("Modelo guardado en 'vgg16_cifar10_mps.pth'")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss por Época")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
for ckpt in checkpoints:
    plt.axvline(ckpt, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy por Época")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("benchmark_resultados.png")
plt.show()
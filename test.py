import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Configurar dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Clases de CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Transformación
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Dataset test
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Definición del modelo (misma arquitectura que antes)
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Cargar modelo
model = VGG16().to(device)
model.load_state_dict(torch.load("best_vgg16_model.pth", map_location=device))
model.eval()

# Seleccionar imágenes de prueba
images, labels = zip(*[test_dataset[i] for i in range(16)])
inputs = torch.stack(images).to(device)

# Predicción
with torch.no_grad():
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)
    top3_probs, top3_idxs = torch.topk(probs, 3, dim=1)

# Mostrar resultados
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = images[i].permute(1, 2, 0).numpy()
    img = (img * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.axis("off")

    text = ""
    for j in range(3):
        idx = top3_idxs[i][j].item()
        prob = top3_probs[i][j].item() * 100
        color = "green" if idx == labels[i] else "darkred"
        text += f"{classes[idx]}: {prob:.1f}%\n"

    ax.set_title(text.strip(), fontsize=10, loc="left")

plt.tight_layout()
plt.savefig("predicciones_grid.png")
plt.show()

print("✅ Imagen guardada como 'predicciones_grid.png'")
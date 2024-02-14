import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import os

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bottleneck = 2


# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Load the Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Initialize the model, criterion, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Autoencoder
num_epochs = 5
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1).to(device)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")

# Create a directory to save the images
os.makedirs('output_images', exist_ok=True)

# Test the network on the test data
dataiter = iter(test_loader)
images, _ = next(dataiter)  # Corrected line

# Pass the test images through the autoencoder
images = images.view(images.size(0), -1).to(device)
outputs = model(images)
outputs = outputs.view(10, 1, 28, 28).cpu()

# Save original and reconstructed images
for idx in range(images.size(0)):
    # Save original image
    torchvision.utils.save_image(images[idx].view(1, 28, 28), f'output_images/original_{idx}.png')

    # Save reconstructed image
    torchvision.utils.save_image(outputs[idx], f'output_images/reconstructed_{idx}.png')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import *
from dataset import *

# Step 1: Prepare Dataset and DataLoader for a single 3D object
train_dataset = OccupancyDataset(file_path="./processed/utah_teapot.obj")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 2: Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OccupancyModel(grid_size=32, feature_size=16, hidden_size=64).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()  # Set the model to training mode

num_epochs = 20

# Step 3: Training Loop
for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as pbar:  # Wrap the train_loader with tqdm for a progress bar
        for points, labels in pbar:
            pbar.set_description(f"Epoch {epoch + 1}")
            
            points, labels = points.to(device), labels.to(device)
            labels = labels.view(-1, 1)  # Ensure labels have the correct shape
            
            optimizer.zero_grad()  # Zero the gradients
            
            outputs = model(points)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / len(train_loader))

# Save the trained model
model_path = os.path.join(os.getcwd(), "utah_teapot_occupancy_model_100e_singleLoD.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
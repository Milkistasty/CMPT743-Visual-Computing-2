import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import *
from dataset import *

# Path to the directory containing object files
directory = "./processed/"

# Hyperparameters
batch_size = 4096
learning_rate = 0.01
num_epochs = 100

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over object files in the directory
for obj_file in os.listdir(directory):
    print(f"Training model for {obj_file}...")

    # Prepare Dataset and DataLoader for the current object
    dataset_name = os.path.splitext(obj_file)[0]
    train_dataset = OccupancyDataset(file_path=os.path.join(directory, obj_file))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    # Single LoD
    # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=8, num_lods=1, mlp_hidden_dim=64, num_layers=2).to(device)
    # Multi LoD
    # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=6, num_lods=3, mlp_hidden_dim=64, num_layers=2).to(device)
    # Hash
    model = OCCNet(grid_type="hash", grid_feat_dim=16, base_lod=6, num_lods=3, mlp_hidden_dim=64, num_layers=2).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set the model to training mode

    # Step 3: Training Loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as pbar:  # Wrap the train_loader with tqdm for a progress bar
            for points, labels in pbar:
                pbar.set_description(f"Epoch {epoch + 1}")

                points, labels = points.to(device), labels.to(device)  # Move data to device
                labels = labels.view(-1, 1)  # Ensure labels have the correct shape [batch size] -> [batch size, 1]

                optimizer.zero_grad()  # Zero the gradients

                outputs = model(points)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / len(train_loader))

    # Save the trained model
    file_name = os.path.basename(train_dataset.file_path)
    dataset_name = os.path.splitext(file_name)[0]
    model_path = f"./model/{dataset_name}_occupancy_model_{epoch+1}e_multiHash.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
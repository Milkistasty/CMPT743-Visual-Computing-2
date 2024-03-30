import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import *
from dataset import *


# Path to the directory containing object files
directory = "./processed/"

# Hyperparameters
batch_size = 4096
initial_learning_rate = 0.01
weight_decay = 1e-5
lr_decay_step = 20
lr_decay_factor = 0.1
num_epochs = 200


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over object files in the directory
for obj_file in os.listdir(directory):
    print(f"Training model for {obj_file}...")

    # Prepare Dataset and DataLoader for the current object
    dataset_name = os.path.splitext(obj_file)[0]
    train_dataset = OccupancyDataset(file_path=os.path.join(directory, obj_file))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    # Single LoD
    # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=8, num_lods=1, mlp_hidden_dim=64, num_layers=2).to(device)
    # Multi LoD
    # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=6, num_lods=3, mlp_hidden_dim=64, num_layers=2).to(device)
    # Hash
    model = OCCNet(grid_type="hash", grid_feat_dim=4, base_lod=4, num_lods=6, mlp_hidden_dim=256, num_layers=9).to(device)

    criterion = torch.nn.BCELoss()  # init loss function
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)  # init optimizer with weight decay
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)  # StepLR scheduler

    model.train()  # Set the model to training mode
    
    # Initialize variables to track best loss and corresponding model state
    best_loss = float('inf')
    best_model_state = None

    # Step 3: Training Loop
    for epoch in range(num_epochs):
        total_loss = 0.0
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
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(train_loader))

        # Check if the current epoch's loss is better than the previous best loss
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state = model.state_dict()

    # Save the trained model
    file_name = os.path.basename(train_dataset.file_path)
    dataset_name = os.path.splitext(file_name)[0]
    model_path = f"./model/{dataset_name}_OCCNet_best_model_multiHash.pth"
    torch.save(best_model_state, model_path)
    print(f"Best Model saved to {model_path}")
    print(f"Best Loss is {best_loss/len(train_loader)}")

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class OccupancyDataset(Dataset):
    def __init__(self, file_path):
        """
        Initializes the dataset for training mode with random subsamples from the given object file.
        Args:
            file_path (string): Path to a single object file.
        """
        self.file_path = file_path
        self.data = self._load_data(file_path)
        self.count_occurrences()
        self.downsampling()

    def _load_data(self, file_path):
        """
        Load data from a file, parsing each line as coordinates and occupancy.
        Occupancy is determined by the color: red for empty (-1), green for occupied (1).
        """
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts[0] == 'v' and len(parts) == 7:  # Ensure it's a vertex line with expected number of elements
                    x, y, z = map(float, parts[1:4])  # the first three elements are coordinates
                    r, g, b = map(float, parts[4:7])  # followed by the r, g, b
                    
                    # occupancy value -1 means empty and 1 means occupied
                    # but I changed it to 0 and 1, to utilize BCE loss 

                    # Determine occupancy based on color
                    if r == 1.0 and g == 0.0 and b == 0.0:
                        occupancy = 0  # Red for empty
                    elif g == 1.0 and r == 0.0 and b == 0.0:
                        occupancy = 1  # Green for occupied
                    else:
                        occupancy = 0  # Default case, might adjust based on dataset specifics
                
                    data.append(((x, y, z), occupancy))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point, occupancy = self.data[idx]
        point = torch.tensor(point, dtype=torch.float32)  # Convert to tensor
        occupancy = torch.tensor(occupancy, dtype=torch.float32)  # Convert to tensor
        return point, occupancy
    
    def count_occurrences(self):
        """
        Count occurrences of each occupancy value in the dataset.
        """
        self.occurrences = {
            0: 0,
            1: 0
        }
        for _, occupancy in self.data:
            self.occurrences[occupancy] += 1

    def downsampling(self):
        """
        Balance the dataset by downsampling the majority class to match the size of the minority class.
        """
        # Find the size of the minority class
        minority_size = min(self.occurrences.values())
        
        # Downsample the majority class to match the size of the minority class
        balanced_data = []
        for point, occupancy in self.data:
            if self.occurrences[occupancy] > minority_size:
                if occupancy == 0:
                    if np.random.rand() < minority_size / self.occurrences[occupancy]:
                        balanced_data.append((point, occupancy))
                        self.occurrences[occupancy] -= 1
                else:
                    balanced_data.append((point, occupancy))
                    self.occurrences[occupancy] -= 1
            else:
                balanced_data.append((point, occupancy))
        
        self.data = balanced_data



# Testing

# from torch.utils.data import DataLoader
        
# dataset = OccupancyDataset(directory="./processed/")

# num_samples_to_display = 5

# print(f"Displaying first {num_samples_to_display} samples from the dataset:")
# for i in range(num_samples_to_display):
#     # Retrieve the i-th sample from the dataset
#     sample = dataset[i]

#     print(f"Sample {i}: Coordinates: {sample[0]}, Occupancy: {sample[1]}")

# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# for batch_idx, sample_batched in enumerate(data_loader):
#     print(f"Batch {batch_idx}:")
#     print(f"Coordinates Batch: {sample_batched[0]}")
#     print(f"Occupancy Batch: {sample_batched[1]}")
#     break



# directory = "./processed"

# for obj_file in os.listdir(directory):
#     print(f"Object file: {obj_file}")
#     dataset = OccupancyDataset(os.path.join(directory, obj_file))
#     print("Occurrences:", dataset.occurrences)
#     print()
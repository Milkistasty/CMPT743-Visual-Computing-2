from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))

"""
Single Level of Detail
"""
class FeatureGrid(nn.Module):
    def __init__(self, grid_size=32, feature_size=16):
        """
        Initializes a 3D feature grid.
        Args:
            grid_size (int): The size of the grid (number of nodes along each axis).
            feature_size (int): The size of the feature vector at each grid node.
        """
        super(FeatureGrid, self).__init__()
        self.grid_size = grid_size
        self.feature_size = feature_size
        self.grid = nn.Parameter(torch.randn(grid_size, grid_size, grid_size, feature_size))

    def forward(self, points):
        """
        For each point, finds the nearest grid node and returns its feature vector.
        Args:
            points (tensor): A (N, 3) tensor of points where N is the number of points.
        Returns:
            Tensor of shape (N, feature_size) containing feature vectors for each point.
        """
        # Scale and clamp points to be within the grid indices
        points = points * (self.grid_size - 1)
        points = torch.clamp(points, 0, self.grid_size - 1)
        
        # Find nearest grid indices (simple rounding for nearest neighbor)
        indices = torch.round(points).long()
        
        # Gather the features from the grid
        features = self.grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        return features

class OccupancyNet(nn.Module):
    def __init__(self, feature_size=16, hidden_size=64):
        """
        Initializes the MLP for predicting occupancy from feature vectors.
        Args:
            feature_size (int): The size of the input feature vector.
            hidden_size (int): The size of the hidden layers in the MLP.
        """
        super(OccupancyNet, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output occupancy value

    def forward(self, x):
        """
        Forward pass of the MLP.
        Args:
            x (tensor): Input feature vectors.
        Returns:
            Tensor of occupancy values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh to get a value between -1 and 1
        return x

class OccupancyModel(nn.Module):
    def __init__(self, grid_size=32, feature_size=16, hidden_size=64):
        super(OccupancyModel, self).__init__()
        self.feature_grid = FeatureGrid(grid_size, feature_size)
        self.occupancy_net = OccupancyNet(feature_size, hidden_size)

    def forward(self, points):
        features = self.feature_grid(points)
        occupancy = self.occupancy_net(features)
        return occupancy
    

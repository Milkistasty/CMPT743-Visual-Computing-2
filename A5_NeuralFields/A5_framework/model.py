from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils import *


class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))

#**************************************************************************************

class DenseGrid(nn.Module):
    def __init__(self, base_lod, num_lods, feat_dim, device='cuda'):
        super(DenseGrid, self).__init__()

        self.LOD = [2 ** l for l in range(base_lod, base_lod + num_lods)]
        self.feat_dim = feat_dim
        self.device = device
        self.feature_grids = nn.ParameterList()
        self.init_feature_grids()

    def init_feature_grids(self):
        for l, lod in enumerate(self.LOD):
            feature_grid = nn.Parameter(torch.zeros((lod ** 3, self.feat_dim), dtype=torch.float32, device=self.device))
            torch.nn.init.normal_(feature_grid, mean=0, std=0.01)
            self.feature_grids.append(feature_grid)

    def forward(self, pts):
        #TODO: Given 3D points, use the bilinear interpolation function in the utils file to interpolate the features from the feature grids
        #TODO: concat interpolated feature from each LoD and return the concatenated tensor
        feats = []
        for l, lod, in enumerate(self.LOD):
            feats.append(bilinear_interp(self.feature_grids[l], pts, res=lod))
        feats = torch.cat(feats, dim=1)
        return feats

#**************************************************************************************

class HashGrid(nn.Module):
    def __init__(self, min_res, max_res, num_lod, hash_bandwidth, feat_dim, device='cuda'):
        super(HashGrid, self).__init__()

        self.min_res = min_res
        self.max_res = max_res
        self.num_lod = num_lod
        self.feat_dim = feat_dim
        self.device = device
        self.hash_table_size = 2 ** hash_bandwidth

        b = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_lod - 1))
        self.LOD = [int(1 + np.floor(self.min_res * (b**l))) for l in range(self.num_lod)]
        self.feature_grids = nn.ParameterList()
        self.init_feature_grids()

    def init_feature_grids(self):
        for l, lod in enumerate(self.LOD):
            feature_grid = nn.Parameter(torch.zeros(min(lod ** 3, self.hash_table_size) , self.feat_dim, dtype=torch.float32, device = self.device))
            torch.nn.init.normal_(feature_grid, mean=0, std=0.001)
            self.feature_grids.append(feature_grid)

    def forward(self, pts):
        #TODO: Given 3D points, use the hash function to interpolate the features from the feature grids
        #TODO: concat interpolated feature from each LoD and return the concatenated tensor
        
        feats = []

        #TODO: add some minor noises to the points; not working
        # pts = pts + torch.randn_like(pts) * 0.00001

        for l, lod, in enumerate(self.LOD):
            feats.append(bilinear_interp(self.feature_grids[l], pts, res=lod, grid_type="hash"))
        feats = torch.cat(feats, dim=1)
        return feats

#**************************************************************************************
    
class MLP(nn.Module):
    def __init__(self, num_mlp_layers, mlp_width, feat_dim, num_lod):
        super(MLP, self).__init__()

        self.num_layers = num_mlp_layers
        self.width = mlp_width

        self.layers = nn.ModuleList()
        input_dim  = feat_dim * num_lod
        
        #TODO: Create a multi-layer perceptron with num_layers layers and width width
        # Create the MLP layers
        self.layers.append(nn.Linear(input_dim, self.width))  # Input layer
        self.layers.append(nn.ReLU())  # Activation function
        for _ in range(num_mlp_layers - 1):
            self.layers.append(nn.Linear(self.width, self.width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.width, 1))  # Output layer
        self.model = nn.Sequential(*self.layers)

        # Initialize the layers with Xavier initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        #x is the concatenated feature tensor from the feature grids
        #TODO: pass x through the MLP and return the output which is p(point is inside the object)
        out = self.model(x)
        return torch.sigmoid(out)

#**************************************************************************************
    
class OCCNet(nn.Module):
    def __init__(self, grid_type="dense", grid_feat_dim=4, base_lod=4, num_lods=6, mlp_hidden_dim=64, num_layers=2):
        #OCCNet is the main model that combines a feature grid (Dense or Hash grid) and an MLP
        #TODO: Initialize the feature grid and MLP
        super(OCCNet, self).__init__()

        if grid_type == 'dense':
            self.dense_grid = DenseGrid(feat_dim=grid_feat_dim, base_lod=base_lod, num_lods=num_lods)
        elif grid_type == 'hash':
            self.dense_grid = HashGrid(min_res=2**base_lod, max_res=2**(base_lod+num_lods-1), num_lod=num_lods, hash_bandwidth=13, feat_dim=grid_feat_dim)
        else:
            raise NotImplementedError('Grid type not implemented')
        
        self.mlp = MLP(num_layers, mlp_hidden_dim, grid_feat_dim, num_lods)

    def get_params(self, lr):
        params = [
            {'params': self.dense_grid.parameters(), 'lr': lr * 10},
            {'params': self.mlp.parameters(), 'lr': lr}
        ]
        return params

    def forward(self, x):
        #check if x is numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().cuda()
        x = self.dense_grid(x)
        x = self.mlp(x)
        return x    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *

class HypergraphLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(HypergraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, H):
        D_v = torch.diag(H.sum(1))
        D_e = torch.diag(H.sum(0))
        
        x = torch.matmul(H, x)
        x = torch.matmul(D_e.inverse(), x)
        x = torch.matmul(H.t(), x)
        x = torch.matmul(D_v.inverse(), x)
        
        return self.linear(x)

class FastKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_min=-3., grid_max=3., num_grids=8):
        super().__init__()
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim)
        self.base_linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        spline_basis = self.rbf(x)
        spline_out = self.spline_linear(spline_basis.view(x.size(0), -1))
        base_out = self.base_linear(F.silu(x))
        return spline_out + base_out

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min=-4., grid_max=4., num_grids=8):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = (grid_max - grid_min) / (num_grids - 1)
        
    def forward(self, x):
        return torch.exp(-((x.unsqueeze(-1) - self.grid) / self.denominator) ** 2)

class SplineLinear(nn.Linear):
    def __init__(self, in_features, out_features, init_scale=0.1):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False)
        
    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class HKCHImageNet(nn.Module):
    def __init__(self, input_dim, bit=128, hidden_dims=[1024, 1024]):
        super().__init__()
        fkan_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            fkan_layers.append(FastKANLayer(prev_dim, hidden_dim))
            fkan_layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        self.fkan = nn.Sequential(*fkan_layers)
        self.hypergraph = HypergraphLayer(prev_dim, bit)
        
    def build_hypergraph(self, features, k=5, threshold=0.6):
        n = features.size(0)
        dists = torch.cdist(features, features)
        H = torch.zeros(n, n, device=features.device)
        
        for i in range(n):
            _, indices = torch.topk(-dists[i], k=k, largest=False)
            H[i, indices] = 1.0
            
        return H
    
    def forward(self, x):
        features = self.fkan(x)
        H = self.build_hypergraph(features)
        hash_codes = self.hypergraph(features, H).tanh()
        return hash_codes

class HKCHTextNet(nn.Module):
    def __init__(self, input_dim, bit=128, hidden_dims=[1024, 1024]):
        super().__init__()
        self.net = HKCHImageNet(input_dim, bit, hidden_dims)
    
    def forward(self, x):
        return self.net(x)
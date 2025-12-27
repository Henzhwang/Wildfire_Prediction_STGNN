"""
Spatiotemporal Graph Neural Network for BC Wildfire Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple


class TemporalAttention(nn.Module):
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        """
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)


class SpatialGraphConv(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 conv_type: str = 'gcn'):
        super(SpatialGraphConv, self).__init__()
        
        if conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'gat':
            self.conv = GATConv(in_channels, out_channels, heads=4, concat=False)
        elif conv_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # self.batch_norm = nn.BatchNorm1d(out_channels)
        self.batch_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.batch_norm(x)
        return F.relu(x)


class STGNNBlock(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 spatial_channels: int,
                 temporal_channels: int,
                 num_nodes: int):
        super(STGNNBlock, self).__init__()
        
        # time conv
        self.temporal_conv1 = nn.Conv2d(
            in_channels, temporal_channels,
            kernel_size=(1, 3), padding=(0, 1)
        )
        
        # spatial conv
        self.spatial_conv = SpatialGraphConv(
            temporal_channels, spatial_channels, conv_type='gcn'
        )
        
        # time conv 2
        self.temporal_conv2 = nn.Conv2d(
            spatial_channels, in_channels,
            kernel_size=(1, 3), padding=(0, 1)
        )
        
        self.batch_norm = nn.BatchNorm2d(in_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch_size, in_channels, num_nodes, seq_len]
        """
        batch_size, _, num_nodes, seq_len = x.shape
        
        # time conv 1
        residual = x
        x = F.relu(self.temporal_conv1(x))  # [B, temporal_channels, N, T]
        
        # spatial conv
        # reshape to [B*T, N, C]
        x = x.permute(0, 3, 2, 1)  # [B, T, N, C]
        x = x.reshape(batch_size * seq_len, num_nodes, -1)  # [B*T, N, C]
        
        x = self.spatial_conv(x, edge_index, edge_weight)  # [B*T, N, spatial_channels]
        
        # reshape back to [B, C, N, T]
        x = x.reshape(batch_size, seq_len, num_nodes, -1)  # [B, T, N, C]
        x = x.permute(0, 3, 2, 1)  # [B, C, N, T]
        
        # time conv 2
        x = self.temporal_conv2(x)  # [B, in_channels, N, T]
        
        # residual
        x = self.batch_norm(x + residual)
        
        return x


class BCWildfireSTGNN(nn.Module):
    
    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 hidden_dim: int = 64,
                 num_stgnn_blocks: int = 3,
                 dropout: float = 0.3,
                 use_temporal_attention: bool = True):


        super(BCWildfireSTGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.use_temporal_attention = use_temporal_attention
        
        # input embedding
        self.input_embedding = nn.Linear(num_features, hidden_dim)
        
        # ST-GNN blocks
        self.stgnn_blocks = nn.ModuleList([
            STGNNBlock(
                in_channels=hidden_dim,
                spatial_channels=hidden_dim,
                temporal_channels=hidden_dim,
                num_nodes=num_nodes
            ) for _ in range(num_stgnn_blocks)
        ])
        
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim, num_heads=4)
        
        # gru for spatiotemporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_stgnn_blocks > 1 else 0
        )
        
        # output layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        """        
        Returns:
            output: [batch_size, num_nodes, 1] - fire prob at each node
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # input embedding: [B, T, N, F] -> [B, T, N, H]
        x = self.input_embedding(x)
        
        # transform to [B, H, N, T] for ST-GNN
        x = x.permute(0, 3, 2, 1)
        
        # ST-GNN blocks
        for stgnn_block in self.stgnn_blocks:
            x = stgnn_block(x, edge_index, edge_weight)
        
        # reshape back to [B, T, N, H]
        x = x.permute(0, 3, 2, 1)
        
        if self.use_temporal_attention:
            # reshape to [B*N, T, H]
            x = x.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
            x = self.temporal_attention(x)
            x = x.reshape(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # GRU temporal modeling: [B*N, T, H] -> [B*N, T, H]
        x = x.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
        x, _ = self.gru(x)
        
        # last time step
        x = x[:, -1, :]  # [B*N, H]
        
        # output layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [B*N, 1]
        
        # reshape to [B, N, 1]
        x = x.reshape(batch_size, num_nodes, 1)
        
        return x


class SimpleSTGNN(nn.Module):
    """
    For testing
    """
    
    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 hidden_dim: int = 64,
                 num_gcn_layers: int = 2,
                 dropout: float = 0.3):
        super(SimpleSTGNN, self).__init__()
        
        self.num_nodes = num_nodes
        
        # input embedding
        self.input_fc = nn.Linear(num_features, hidden_dim)
        
        # gcn layers
        self.gcn_layers = nn.ModuleList([
            SpatialGraphConv(hidden_dim, hidden_dim, conv_type='gcn')
            for _ in range(num_gcn_layers)
        ])
        
        # LSTM for time series
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # output layer
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, num_features]
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # [B, T, N, F] -> [B, T, N, H]
        x = F.relu(self.input_fc(x))
        
        # apply gcn to each time step
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]  # [B, N, H]
            xt = xt.reshape(batch_size * num_nodes, -1)  # [B*N, H]
            

            for gcn_layer in self.gcn_layers:
                xt = gcn_layer(xt, edge_index, edge_weight)
            
            xt = xt.reshape(batch_size, num_nodes, -1)  # [B, N, H]
            gcn_outputs.append(xt)
        
        # [B, T, N, H]
        x = torch.stack(gcn_outputs, dim=1)
        
        # [B*N, T, H] -> [B*N, T, H]
        x = x.reshape(batch_size * num_nodes, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last timestep [B*N, H]
        
        # predict
        x = self.fc_out(x)  # [B*N, 1]
        x = x.reshape(batch_size, num_nodes, 1)
        
        return x


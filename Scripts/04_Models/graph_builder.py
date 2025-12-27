"""
Graph Builder for BC Wildfire Spatial Network
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional
import pickle


class BCFireGraphBuilder:
    """
    Graph builder
    Transform 2,367 Grids to GNN nodes
    Proximity as Edges
    """
    
    def __init__(self, 
                 num_neighbors: int = 8,
                 distance_threshold: float = 0.35):
        
        self.num_neighbors = num_neighbors
        self.distance_threshold = distance_threshold
        self.grid_to_node = None
        self.node_to_grid = None
        self.edge_index = None
        self.edge_attr = None
        
    def build_graph_from_data(self, 
                             df: pd.DataFrame,
                             save_path: Optional[str] = None) -> Dict:
        """
            
        Returns:
            dict with graph strcture
        """
        # obtain only grid
        grid_info = df[['grid_id', 'centroid_lon', 'centroid_lat']].drop_duplicates()
        grid_info = grid_info.sort_values('grid_id').reset_index(drop=True)
        
        num_nodes = len(grid_info)
        print(f"{num_nodes} nodes")
        
        
        self.grid_to_node = {gid: idx for idx, gid in enumerate(grid_info['grid_id'])}
        self.node_to_grid = {idx: gid for gid, idx in self.grid_to_node.items()}
        
        # extract coordinates
        coords = grid_info[['centroid_lon', 'centroid_lat']].values
        # transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
        # x, y = transformer.transform(grid_info['centroid_lon'].values, grid_info['centroid_lat'].values)
        # coords = np.stack([x, y], axis=1)
        
        # build KD tree
        kdtree = KDTree(coords)
        
        # edges
        edge_list = []
        edge_distances = []
        
        for node_idx in range(num_nodes):
            # find K-nearest neighbors
            distances, neighbors = kdtree.query(
                coords[node_idx], 
                k=self.num_neighbors + 1  
            )
            
            # exclude itself but real neighbor
            for neighbor_idx, dist in zip(neighbors[1:], distances[1:]):
                if dist <= self.distance_threshold:
                    edge_list.append([node_idx, neighbor_idx])
                    edge_distances.append(dist)
        # print(edge_list)
        # print(len(edge_list))
        
        # transform PyTorch Geometric
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float32).view(-1, 1)
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        # dictionary
        graph_structure = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.shape[1],
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'grid_to_node': self.grid_to_node,
            'node_to_grid': self.node_to_grid,
            'coordinates': coords,
            'grid_info': grid_info
        }
        
        # Graph Strcture
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(graph_structure, f)
        
        # stats
        avg_degree = edge_index.shape[1] / num_nodes
        print(f"#Edges: {edge_index.shape[1]:,}")
        print(f"Avg degree: {avg_degree:.2f}")
        print(f"Range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
        
        return graph_structure
    
    @staticmethod
    def load_graph_structure(path: str) -> Dict:
        
        with open(path, 'rb') as f:
            graph_structure = pickle.load(f)
        print(f"Loaded Graph: {graph_structure['num_nodes']} nodes, "
              f"{graph_structure['num_edges']} edges")
        return graph_structure
    
    def visualize_graph(self, graph_structure: Dict, sample_nodes: int = 100):

        import matplotlib.pyplot as plt
        
        coords = graph_structure['coordinates']
        edge_index = graph_structure['edge_index'].numpy()
        
        # sample nodes
        if sample_nodes < len(coords):
            sample_idx = np.random.choice(len(coords), sample_nodes, replace=False)
            sample_mask = np.isin(edge_index[0], sample_idx) & np.isin(edge_index[1], sample_idx)
            sample_edges = edge_index[:, sample_mask]
        else:
            sample_idx = np.arange(len(coords))
            sample_edges = edge_index
        
        # plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # edges
        for i in range(sample_edges.shape[1]):
            src, dst = sample_edges[:, i]
            ax.plot([coords[src, 0], coords[dst, 0]], 
                   [coords[src, 1], coords[dst, 1]], 
                   'b-', alpha=0.2, linewidth=0.5)
        
        # nodes
        ax.scatter(coords[sample_idx, 0], coords[sample_idx, 1], 
                  c='red', s=20, alpha=0.6, zorder=5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Spatial Graph Strcture (sampled {sample_nodes} nodes)')
        ax.grid(True, alpha=0.3)
        
        return fig


def create_temporal_graph_snapshot(df: pd.DataFrame,
                                   date: str,
                                   graph_structure: Dict,
                                   feature_cols: List[str],
                                   target_col: str = 'Fire_occurred') -> Data:
    """
    Returns:
        PyTorch Geometric Data objects
    """
    # filter data from specific dates
    date_df = df[df['date'] == date].copy()
    
    # sort
    date_df = date_df.sort_values('grid_id').reset_index(drop=True)
    
    # extract features and labels
    x = torch.tensor(date_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(date_df[target_col].values, dtype=torch.float32)
    
    # graph strcture object
    data = Data(
        x=x,
        edge_index=graph_structure['edge_index'],
        edge_attr=graph_structure['edge_attr'],
        y=y
    )
    
    return data
    

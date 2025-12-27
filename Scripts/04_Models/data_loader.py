"""
Spatiotemporal Graph Data Loader
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta


class STGraphDataset(Dataset):

    
    def __init__(self,
                 df: pd.DataFrame,
                 graph_structure: Dict,
                 feature_cols: List[str],
                 target_col: str = 'Fire_occurred',
                 seq_len: int = 7,
                 forecast_horizon: int = 1,
                 stride: int = 1):
        """
        Args:
            graph_structure: spatial graph
            seq_len: input sequence length
            forecast_horizon: predict how many days
        """
        self.df = df.copy()
        self.graph_structure = graph_structure
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        self.num_nodes = graph_structure['num_nodes']
        self.grid_to_node = graph_structure['grid_to_node']
        
        # obtain unqiue dates
        self.dates = sorted(df['Date'].unique())
        self.date_to_idx = {date: idx for idx, date in enumerate(self.dates)}
        
        
        self.date_to_indices = {}
        self.date_to_features = {}
        self.date_to_targets = {}
        
        for date in self.dates:
            # obtained data for one date
            mask = self.df['Date'] == date
            date_data = self.df[mask].copy()
            
            # sort 
            date_data = date_data.sort_values('grid_id')
        
            
            self.date_to_features[date] = date_data[self.feature_cols].values.astype(np.float32)
            
            self.date_to_targets[date] = date_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        
        
        self.valid_windows = self._create_windows()

        
    def _create_windows(self) -> List[Tuple[int, int]]:
        """
        Returns:
            List of (start_idx, target_idx) tuples
        """
        windows = []
        max_idx = len(self.dates) - self.forecast_horizon
        
        for start_idx in range(0, max_idx - self.seq_len + 1, self.stride):
            target_idx = start_idx + self.seq_len + self.forecast_horizon - 1
            if target_idx < len(self.dates):
                windows.append((start_idx, target_idx))
        
        return windows
    
    def __len__(self):
        return len(self.valid_windows)
    
    def __getitem__(self, idx):
        """
        get a sample
        
        Returns:
            x: [seq_len, num_nodes, num_features]
            y: [num_nodes, 1]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1]
        """
        start_idx, target_idx = self.valid_windows[idx]
        
        # obtained iput dates
        input_dates = self.dates[start_idx:start_idx + self.seq_len]
        target_date = self.dates[target_idx]
        
        # x_list = []
        # for date in input_dates:
        #     date_df = self.df[self.df['Date'] == date].copy()
        #     date_df = date_df.sort_values('grid_id')
        #     x_list.append(date_df[self.feature_cols].values)

        x_list = []
        for date in input_dates:
            # from data
            features = self.date_to_features[date]
            x_list.append(features)
        
        x = np.stack(x_list, axis=0).astype(np.float32)  # [seq_len, num_nodes, num_features]
        
        # stack to  [seq_len, num_nodes, num_features]
        x = np.stack(x_list, axis=0)
        
        y = self.date_to_targets[target_date]

        # target_df = self.df[self.df['Date'] == target_date].copy()
        # target_df = target_df.sort_values('grid_id')
        # y = target_df[self.target_col].values.reshape(-1, 1)  # [num_nodes, 1]
        
        # Xdf = target_df[self.feature_cols]
        # if not hasattr(self, "_printed_debug"):
        #     self._printed_debug = True
        #     print("DEBUG dtypes:\n", Xdf.dtypes)
        #     bad = Xdf.columns[~Xdf.dtypes.apply(lambda t: pd.api.types.is_numeric_dtype(t))]
        #     print("DEBUG non-numeric cols:", list(bad))
        #     print("DEBUG sample values:\n", Xdf[bad].head(3) if len(bad) else "none")

        # if isinstance(x, (pd.Series, pd.DataFrame)):
        #     arr = x.to_numpy()
        # else:
        #     arr = np.asarray(x)

        # if arr.dtype == object:
        #     print("object dtype found!")
        #     print("index:", idx)
        #     if isinstance(x, pd.Series):
        #         bad = x[x.map(lambda v: isinstance(v, (str, object)) and v is not None and not isinstance(v, (int, float, np.number, bool)))]
        #         print("bad entries:", bad)
        #     print("sample:", arr[:10])
        
        # transform to Tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return {
            'x': x,
            'y': y,
            'edge_index': self.graph_structure['edge_index'],
            'edge_attr': self.graph_structure['edge_attr'],
            'Date': target_date
        }


def collate_fn(batch):
    """
    combined sample to a bulk
    """
    x = torch.stack([item['x'] for item in batch])  # [batch, seq_len, nodes, features]
    y = torch.stack([item['y'] for item in batch])  # [batch, nodes, 1]
    edge_index = batch[0]['edge_index']  # all samples share the same graph
    edge_attr = batch[0]['edge_attr']
    dates = [item['Date'] for item in batch]
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'dates': dates
    }


class TemporalSplit:
    """
    split by train valid test
    """
    
    @staticmethod
    def split_by_date(df: pd.DataFrame,
                     train_end: str,
                     val_end: str,
                     test_start: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            train_df, val_df, test_df
        """
        train_df = df[df['Date'] <= train_end].copy()
        val_df = df[(df['Date'] > train_end) & (df['Date'] <= val_end)].copy()
        
        if test_start:
            test_df = df[df['Date'] >= test_start].copy()
        else:
            test_df = df[df['Date'] > val_end].copy()
        
        print(f"   Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df):,} observations)")
        print(f"   Validation: {val_df['Date'].min()} to {val_df['Date'].max()} ({len(val_df):,} observations)")
        print(f"   Test: {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df):,} observations)")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def split_by_ratio(df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        split by ratio
            
        Returns:
            train_df, val_df, test_df
        """
        dates = sorted(df['Date'].unique())
        n_dates = len(dates)
        
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))
        
        train_end = dates[train_end_idx - 1]
        val_end = dates[val_end_idx - 1]
        
        return TemporalSplit.split_by_date(df, train_end, val_end)

def create_weighted_sampler(dataset):
    all_labels = []
    for i in range(len(dataset)):
        y = dataset[i]['y'].numpy()
        all_labels.append(y.flatten())
    
    all_labels = np.concatenate(all_labels)
    
    # compute sample weighting
    class_counts = np.bincount(all_labels.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[all_labels.astype(int)]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      graph_structure: Dict,
                      feature_cols: List[str],
                      target_col: str = 'Fire_occurred',
                      seq_len: int = 7,
                      batch_size: int = 32,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns:
        train_loader, val_loader, test_loader
    """
    
    train_dataset = STGraphDataset(
        train_df, graph_structure, feature_cols, target_col, seq_len
    )
    val_dataset = STGraphDataset(
        val_df, graph_structure, feature_cols, target_col, seq_len
    )
    test_dataset = STGraphDataset(
        test_df, graph_structure, feature_cols, target_col, seq_len
    )
    
    # create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"   Train: {len(train_loader)}")
    print(f"   Validate: {len(val_loader)}")
    print(f"   Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    np.random.seed(42)
    
    dates = pd.date_range('2019-01-01', '2019-01-31', freq='D')
    n_grids = 2367
    
    data = []
    for date in dates:
        for grid_id in range(n_grids):
            data.append({
                'grid_id': grid_id,
                'Date': date.strftime('%Y-%m-%d'),
                'temperature': np.random.randn(),
                'precipitation': np.random.rand(),
                'Fire_occurred': np.random.randint(0, 2)
            })
    
    df = pd.DataFrame(data)
    
    # build graph
    from graph_builder import BCFireGraphBuilder
    builder = BCFireGraphBuilder()
    graph_structure = builder.build_graph_from_data(df)
    
    # split data
    train_df, val_df, test_df = TemporalSplit.split_by_ratio(df, 0.7, 0.15)
    
    feature_cols = ['temperature', 'precipitation']
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        graph_structure,
        feature_cols,
        seq_len=7,
        batch_size=4
    )
    
    batch = next(iter(train_loader))

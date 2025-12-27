"""
Main Script for BC Wildfire Spatiotemporal GNN
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from graph_builder import BCFireGraphBuilder
from stgnn_model import BCWildfireSTGNN, SimpleSTGNN
from data_loader import TemporalSplit, create_dataloaders
from trainer import STGNNTrainer


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


## Paths
PROJECT_ROOT = Path().resolve()
PROJECT_ROOT = PROJECT_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT/'Processed Data'/'grid_all_neighbors.parquet'
# DATA_DIR = PROJECT_ROOT/'Processed Data'/'grid_all.parquet'
OUTPUT_DIR = PROJECT_ROOT/'Output'/'Actual Run'


def load_and_prepare_data(data_path: str, 
                         sample_ratio: float = 1.0) -> pd.DataFrame:
    
    print(f"Load Data: {data_path}")
    
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    
    
    if sample_ratio < 1.0:
        dates = sorted(df['Date'].unique())
        n_dates = int(len(dates) * sample_ratio)
        sampled_dates = dates[:n_dates]
        df = df[df['Date'].isin(sampled_dates)]
    
    print(f"Load complete:")
    print(f"   # observations: {len(df):,}")
    print(f"   Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   # Grids: {df['grid_id'].nunique()}")
    print(f"   # Features: {len(df.columns) - 3}")  # not include grid_id, Date, Fire_occurred
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:


    exclude_cols = ['grid_id', 'Date', 'Fire_occurred',
                    'centroid_lon', 'centroid_lat']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n# Features ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"   {i}. {col}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} features")
    
    return feature_cols


def train_model(args):
    """train model"""
    
    print("\n" + "="*70)
    print("Model Train")
    print("="*70)
    
    # Load
    df = load_and_prepare_data(args.data_path, args.sample_ratio)
    feature_cols = get_feature_columns(df)
    
    # graph structure
    print("\nBuilding graph...")
    graph_builder = BCFireGraphBuilder(
        num_neighbors=args.num_neighbors,
        distance_threshold=args.distance_threshold
    )
    
    graph_structure_path = Path(args.save_dir) / 'graph_structure.pkl'
    if graph_structure_path.exists() and not args.rebuild_graph:
        print(f"   Load Graph: {graph_structure_path}")
        graph_structure = graph_builder.load_graph_structure(str(graph_structure_path))
    else:
        graph_structure = graph_builder.build_graph_from_data(
            df, 
            save_path=str(graph_structure_path)
        )
    
    # data split
    print("\nSpliting data...")
    if args.split_by_date:
        train_df, val_df, test_df = TemporalSplit.split_by_date(
            df, 
            train_end=args.train_end,
            val_end=args.val_end
        )
    else:
        train_df, val_df, test_df = TemporalSplit.split_by_ratio(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
    
    # create data loader
    print("\nCreating data loader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        graph_structure,
        feature_cols,
        target_col='Fire_occurred',
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # create model
    print("\ncreate models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    if args.model_type == 'full':
        model = BCWildfireSTGNN(
            num_features=len(feature_cols),
            num_nodes=graph_structure['num_nodes'],
            hidden_dim=args.hidden_dim,
            num_stgnn_blocks=args.num_stgnn_blocks,
            dropout=args.dropout,
            use_temporal_attention=args.use_attention
        )
    else:  # simple
        model = SimpleSTGNN(
            num_features=len(feature_cols),
            num_nodes=graph_structure['num_nodes'],
            hidden_dim=args.hidden_dim,
            num_gcn_layers=args.num_gcn_layers,
            dropout=args.dropout
        )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   # Hyperparameters: {num_params:,}")
    
    # create trainer
    print("\nCreating Trainer...")
    trainer = STGNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=str(device),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        save_dir=args.save_dir
    )
    
    # train
    print("\n" + "="*70)
    test_metrics = trainer.train(
        num_epochs=args.num_epochs,
        threshold=args.prediction_threshold,
        early_stopping_patience=args.early_stopping_patience,
        save_best=True
    )
    
    # train curves
    print("\nCreating train curves...")
    curve_path = Path(args.save_dir) / 'training_curves.png'
    trainer.plot_training_curves(save_path=str(curve_path))
    
    print("\n" + "="*70)
    print("Train complete!")
    print("="*70)
    
    return trainer, test_metrics


def predict(args):
    
    print("\n" + "="*70)
    print("")
    print("="*70)
    
    # Load Data
    df = load_and_prepare_data(args.data_path, sample_ratio=1.0)
    feature_cols = get_feature_columns(df)
    
    # Load graph
    graph_structure_path = Path(args.save_dir) / 'graph_structure.pkl'
    graph_builder = BCFireGraphBuilder()
    graph_structure = graph_builder.load_graph_structure(str(graph_structure_path))
    
    # only use test
    # _, _, test_df = TemporalSplit.split_by_ratio(df, 0.7, 0.15)
    if args.split_by_date:
        _, _, test_df = TemporalSplit.split_by_date(
            df, 
            train_end=args.train_end,
            val_end=args.val_end
        )
    else:
        _, _, test_df = TemporalSplit.split_by_ratio(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )

    _, _, test_loader = create_dataloaders(
        test_df, test_df, test_df,  # only use test_df
        graph_structure,
        feature_cols,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'full':
        model = BCWildfireSTGNN(
            num_features=len(feature_cols),
            num_nodes=graph_structure['num_nodes'],
            hidden_dim=args.hidden_dim,
            num_stgnn_blocks=args.num_stgnn_blocks,
            dropout=args.dropout,
            use_temporal_attention=args.use_attention
        )
    else:
        model = SimpleSTGNN(
            num_features=len(feature_cols),
            num_nodes=graph_structure['num_nodes'],
            hidden_dim=args.hidden_dim
        )
    
    # load weightin
    checkpoint_path = Path(args.save_dir) / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model Loaded: {checkpoint_path}")
    
    
    print("\nStart predicting...")
    all_predictions = []
    all_probabilities = []
    all_dates = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device) if batch['edge_attr'] is not None else None
            
            outputs = model(x, edge_index, edge_attr)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.1).astype(int)
            
            all_predictions.append(preds)
            all_probabilities.append(probs)
            all_dates.extend(batch['dates'])
    
    # concat results
    predictions = np.concatenate(all_predictions, axis=0)  # [num_samples, num_nodes, 1]
    probabilities = np.concatenate(all_probabilities, axis=0)
    
    print(f"Predict Complete:")
    print(f"   # Predictions: {len(predictions)}")
    print(f"   Predicted date: {len(set(all_dates))}")
    
    # save
    output_path = Path(args.save_dir) / 'predictions.csv'
    
    results = []
    for i, date in enumerate(all_dates):
        for node_idx in range(graph_structure['num_nodes']):
            grid_id = graph_structure['node_to_grid'][node_idx]
            results.append({
                'Date': date,
                'grid_id': grid_id,
                'fire_probability': probabilities[i, node_idx, 0],
                'fire_prediction': predictions[i, node_idx, 0]
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='STGNN for BC Wildfire')
    
    
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict'])
    

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    
    
    parser.add_argument('--num_neighbors', type=int, default=8)
    parser.add_argument('--distance_threshold', type=float, default=0.15)
    parser.add_argument('--rebuild_graph', action='store_true')
    
    
    parser.add_argument('--split_by_date', action='store_true')
    parser.add_argument('--train_end', type=str, default='2022-12-31')
    parser.add_argument('--val_end', type=str, default='2023-12-31')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    
    
    parser.add_argument('--model_type', type=str, default='full',
                       choices=['full', 'simple'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_stgnn_blocks', type=int, default=3)
    parser.add_argument('--num_gcn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_attention', action='store_true')
    
    
    parser.add_argument('--prediction_threshold',
                        type=float,
                        default=0.15)
    parser.add_argument('--seq_len', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['focal', 'weighted_bce', 'bce'])
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        trainer, metrics = train_model(args)
    else:
        results = predict(args)


if __name__ == "__main__":
    main()

# usage: main.py [-h] [--mode {train,predict}] --data_path DATA_PATH
#                [--save_dir SAVE_DIR] [--sample_ratio SAMPLE_RATIO]
#                [--num_neighbors NUM_NEIGHBORS]
#                [--distance_threshold DISTANCE_THRESHOLD] [--rebuild_graph]
#                [--split_by_date] [--train_end TRAIN_END] [--val_end VAL_END]
#                [--train_ratio TRAIN_RATIO] [--val_ratio VAL_RATIO]
#                [--model_type {full,simple}] [--hidden_dim HIDDEN_DIM]
#                [--num_stgnn_blocks NUM_STGNN_BLOCKS]
#                [--num_gcn_layers NUM_GCN_LAYERS] [--dropout DROPOUT]
#                [--use_attention] [--seq_len SEQ_LEN] [--batch_size BATCH_SIZE]
#                [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE]
#                [--weight_decay WEIGHT_DECAY]
#                [--loss_type {focal,weighted_bce,bce}]
#                [--pos_weight POS_WEIGHT]
#                [--early_stopping_patience EARLY_STOPPING_PATIENCE]
#                [--num_workers NUM_WORKERS]

# python main.py \
#   --mode train \
#   --data_path '/Users/henzhwang/Desktop/STA4101/Processed Data/grid_all_neighbors.parquet' \
#   --save_dir '/Users/henzhwang/Desktop/STA4101/Output/Model Run' \
#   --split_by_date --train_end 2022-12-31 --val_end 2023-12-31 \
#   --num_neighbors 8 --distance_threshold 0.35 --rebuild_graph \
#   --model_type simple \
#   --num_gcn_layers 2 \
#   --hidden_dim 64 \
#   --dropout 0.3 \
#   --seq_len 14 \
#   --batch_size 16 \
#   --num_epochs 1 \
#   --learning_rate 0.001 
#   --weight_decay 1e-5 \
#   --loss_type focal \
#   --num_workers 4

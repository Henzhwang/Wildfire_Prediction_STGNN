"""
Training Script for Spatiotemporal GNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_nodes, 1]
            targets: [batch_size, num_nodes, 1]
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


class WeightedBCELoss(nn.Module):
    
    def __init__(self, pos_weight: float = 10.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, inputs, targets):
        return nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight.to(inputs.device)
        )


class Metrics:
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_prob: np.ndarray) -> Dict[str, float]:
        """
        Returns:
            metrics dict
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'auc_pr': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):


        print(f"\n{prefix}metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
        print(f"  Conf Mat: TP={metrics['true_positives']}, "
              f"FP={metrics['false_positives']}, "
              f"TN={metrics['true_negatives']}, "
              f"FN={metrics['false_negatives']}")


class EarlyStopping:
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        """
        Check if early stopping needed
        
        Args:
            val_score
            
        Returns:
            if needed early stopping
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class STGNNTrainer:
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 loss_type: str = 'focal',
                 pos_weight: float = 10.0,
                 save_dir: str = 'checkpoints'):
        """
        Args:
            weight_decay: L2 regularization coefficient
            loss_type: Loss function type ('focal', 'weighted_bce', 'bce')
            pos_weight: Positive class weight (used for weighted binary cross-entropy)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        pos_ratio = 0.015
        pos_weight = (1 - pos_ratio) / pos_ratio
        alpha = 0.9
        gamma = 2.0

        # loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        
        # optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # learning rate
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # train history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc_roc': [],
            'learning_rate': []
        }
        
    def train_epoch(self) -> float:

        """train one epoch"""

        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # transfer to device
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device) if batch['edge_attr'] is not None else None
            
            # forward
            self.optimizer.zero_grad()
            outputs = self.model(x, edge_index, edge_attr)
            
            # compute loss
            loss = self.criterion(outputs, y)
            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, data_loader, threshold=0.15) -> Tuple[float, Dict[str, float]]:


        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        for batch in data_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device) if batch['edge_attr'] is not None else None
            
            # forward
            outputs = self.model(x, edge_index, edge_attr)
            
            # compute loss
            loss = self.criterion(outputs, y)
            total_loss += loss.item()
            num_batches += 1
            
            # collect pred and targets
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            targets = y.cpu().numpy()
            
            all_probs.append(probs.flatten())
            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())
        
        # compute average loss
        avg_loss = total_loss / num_batches
        
        # compute metrics
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = Metrics.calculate_metrics(all_targets, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def train(self, 
             num_epochs: int = 100,
             early_stopping_patience: int = 15,
             threshold=0.15,
             save_best: bool = True):

        print(f"Start Training...")
        print(f"   Device: {self.device}")
        print(f"   Train epoch: {num_epochs}")
        print(f"   Early Stopping Patience: {early_stopping_patience}")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_f1 = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # train
            train_loss = self.train_epoch()
            
            # validate
            val_loss, val_metrics = self.validate(self.val_loader)
            
            # update learning rate
            self.scheduler.step(val_metrics['f1'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc_roc'].append(val_metrics['auc_roc'])
            self.history['learning_rate'].append(current_lr)
            
            # print
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val F1:     {val_metrics['f1']:.4f}")
            print(f"   Val AUC:    {val_metrics['auc_roc']:.4f}")
            print(f"   LR:         {current_lr:.6f}")
            
            # save best model
            if save_best and val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_checkpoint(
                    epoch + 1,
                    val_metrics,
                    filename='best_model.pth'
                )
                print(f"   Save best model (F1={best_val_f1:.4f})")
            
            # check early stopping
            if early_stopping(val_metrics['f1']):
                print(f"\nEarly Stopping Active (patience={early_stopping_patience})")
                break
        
        # evaluation
        print(f"\Final Evaluation...")
        test_loss, test_metrics = self.validate(self.test_loader)
        Metrics.print_metrics(test_metrics, prefix="test set")
        
        # save train history
        self.save_history()
        
        return test_metrics
    


    def save_checkpoint(self, 
                       epoch: int,
                       metrics: Dict[str, float],
                       filename: str = 'checkpoint.pth'):


        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
    


    def load_checkpoint(self, filename: str = 'best_model.pth'):


        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Load checkpoint: {filename}")
        return checkpoint
    


    def save_history(self):


        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Train history saved: {history_file}")
    


    def plot_training_curves(self, save_path: Optional[str] = None):


        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 scores
        axes[0, 1].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Validation F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC-ROC
        axes[1, 0].plot(self.history['val_auc_roc'], label='Val AUC-ROC', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_title('Validation AUC-ROC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # learning rate
        axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Train curves saved: {save_path}")
        
        return fig

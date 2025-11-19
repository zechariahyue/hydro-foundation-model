#!/usr/bin/env python3
"""
Traditional Transfer Learning LSTM Implementation
=====================================

This module implements traditional LSTM transfer learning for direct comparison
with foundation models. Addresses Reviewer #3's critical concern about fair comparison.

Key Features:
1. LSTM pre-training on source basins
2. Traditional fine-tuning on target basins  
3. Standardized hyperparameter protocols
4. Direct performance comparison framework

Author: Zero Water Team
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalTransferLSTM(nn.Module):
    """
    Traditional LSTM with transfer learning capabilities
    
    Architecture:
    - Multi-layer LSTM with standardized hyperparameters
    - Dropout regularization
    - Linear output projection
    - Transfer learning via pre-training + fine-tuning
    """
    
    def __init__(self, 
                 input_size: int = 3,
                 hidden_size: int = 128, 
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store training state
        self.is_pretrained = False
        self.transfer_mode = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            predictions: Output tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use final timestep output
        final_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply dropout and project to output
        output = self.dropout_layer(final_output)
        predictions = self.output_projection(output)
        
        return predictions
    
    def enable_transfer_mode(self, freeze_layers: Optional[List[str]] = None):
        """
        Enable transfer learning mode
        
        Args:
            freeze_layers: List of layer names to freeze during transfer
        """
        self.transfer_mode = True
        
        if freeze_layers:
            for name, param in self.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")


class TraditionalTransferFramework:
    """
    Complete framework for traditional LSTM transfer learning
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize framework with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract relevant configurations
        self.model_config = self.config['models']['baselines']['lstm']
        self.training_config = self.config['training']
        
        # Initialize model
        self.model = TraditionalTransferLSTM(**self.model_config)
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'pretrain': {'loss': [], 'val_loss': []},
            'transfer': {'loss': [], 'val_loss': []}
        }
        
    def prepare_data(self, 
                    data: Dict[str, np.ndarray], 
                    context_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare time series data for LSTM training
        
        Args:
            data: Dictionary with 'features' and 'targets' arrays
            context_length: Length of input sequences
            
        Returns:
            X: Input sequences [num_samples, context_length, num_features]
            y: Target values [num_samples, 1]
        """
        features = data['features']  # [time_steps, num_features]
        targets = data['targets']    # [time_steps,]
        
        # Create sequences
        X, y = [], []
        for i in range(context_length, len(features)):
            X.append(features[i-context_length:i])
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def pretrain_on_source(self, 
                          source_data: Dict[str, Dict[str, np.ndarray]],
                          epochs: int = 50,
                          validate: bool = True) -> Dict[str, float]:
        """
        Pre-train LSTM on source basins (data-rich regions)
        
        Args:
            source_data: Dict of basin_id -> {'features': array, 'targets': array}
            epochs: Number of training epochs
            validate: Whether to use validation split
            
        Returns:
            training_metrics: Final training metrics
        """
        logger.info(f"Starting pre-training on {len(source_data)} source basins")
        
        # Combine all source basin data
        all_features = []
        all_targets = []
        
        for basin_id, basin_data in source_data.items():
            features = basin_data['features']
            targets = basin_data['targets']
            all_features.append(features)
            all_targets.append(targets)
        
        # Concatenate all data
        combined_features = np.vstack(all_features)
        combined_targets = np.hstack(all_targets)
        
        # Prepare sequences
        X, y = self.prepare_data({
            'features': combined_features,
            'targets': combined_targets
        })
        
        # Train/validation split
        if validate:
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training step
            optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.training_config['gradient_clip_norm']
            )
            
            optimizer.step()
            
            # Validation step
            if validate and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val)
                    val_loss = criterion(val_predictions, y_val)
                
                self.model.train()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    torch.save(self.model.state_dict(), 'results/models/best_pretrained_lstm.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Val_Loss={val_loss:.4f}")
                
                # Store history
                self.training_history['pretrain']['loss'].append(loss.item())
                self.training_history['pretrain']['val_loss'].append(val_loss.item())
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss={loss:.4f}")
                self.training_history['pretrain']['loss'].append(loss.item())
        
        # Load best model
        if validate:
            self.model.load_state_dict(torch.load('results/models/best_pretrained_lstm.pth'))
        
        self.model.is_pretrained = True
        logger.info("Pre-training completed successfully")
        
        return {
            'final_train_loss': loss.item(),
            'final_val_loss': val_loss.item() if validate else None,
            'epochs_trained': epoch + 1
        }
    
    def transfer_to_target(self, 
                          target_data: Dict[str, np.ndarray],
                          fine_tune_epochs: int = 20,
                          freeze_layers: Optional[List[str]] = None,
                          learning_rate_factor: float = 0.1) -> Dict[str, float]:
        """
        Fine-tune pre-trained model on target basin
        
        Args:
            target_data: Target basin data {'features': array, 'targets': array}
            fine_tune_epochs: Number of fine-tuning epochs
            freeze_layers: Layers to freeze during fine-tuning
            learning_rate_factor: LR reduction factor for fine-tuning
            
        Returns:
            transfer_metrics: Transfer learning performance metrics
        """
        if not self.model.is_pretrained:
            raise ValueError("Model must be pre-trained before transfer learning")
        
        logger.info("Starting transfer learning to target basin")
        
        # Enable transfer mode
        self.model.enable_transfer_mode(freeze_layers)
        
        # Prepare target data
        X_target, y_target = self.prepare_data(target_data)
        
        # Use only a subset for few-shot learning (simulate data scarcity)
        few_shot_samples = self.training_config.get('few_shot_samples', 50)
        if len(X_target) > few_shot_samples:
            indices = np.random.choice(len(X_target), few_shot_samples, replace=False)
            X_target = X_target[indices]
            y_target = y_target[indices]
        
        # Fine-tuning optimizer with reduced learning rate
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.training_config['learning_rate'] * learning_rate_factor,
            weight_decay=self.training_config['weight_decay']
        )
        
        criterion = nn.MSELoss()
        
        # Fine-tuning loop
        self.model.train()
        
        for epoch in range(fine_tune_epochs):
            optimizer.zero_grad()
            predictions = self.model(X_target)
            loss = criterion(predictions, y_target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.training_config['gradient_clip_norm']
            )
            
            optimizer.step()
            
            if epoch % 5 == 0:
                logger.info(f"Transfer Epoch {epoch}: Loss={loss:.4f}")
            
            self.training_history['transfer']['loss'].append(loss.item())
        
        # Save fine-tuned model
        torch.save(self.model.state_dict(), 'results/models/traditional_transfer_lstm.pth')
        
        logger.info("Transfer learning completed successfully")
        
        return {
            'final_transfer_loss': loss.item(),
            'epochs_transferred': fine_tune_epochs,
            'samples_used': len(X_target)
        }
    
    def evaluate_on_test(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test data {'features': array, 'targets': array}
            
        Returns:
            metrics: Performance metrics
        """
        self.model.eval()
        
        # Prepare test data
        X_test, y_test = self.prepare_data(test_data)
        
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Convert to numpy for metrics calculation
        y_true = y_test.numpy().flatten()
        y_pred = predictions.numpy().flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Nash-Sutcliffe Efficiency
        nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            'MSE': mse,
            'MAE': mae, 
            'RMSE': rmse,
            'NSE': nse,
            'num_samples': len(y_true)
        }
        
        logger.info(f"Test Performance: MSE={mse:.4f}, MAE={mae:.4f}, NSE={nse:.4f}")
        
        return metrics


def create_synthetic_basin_data(num_basins: int = 5, 
                               time_steps: int = 1000,
                               num_features: int = 3) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create synthetic basin data for testing
    
    Args:
        num_basins: Number of synthetic basins
        time_steps: Length of time series
        num_features: Number of input features
        
    Returns:
        basin_data: Dictionary of basin data
    """
    basin_data = {}
    
    for basin_id in range(num_basins):
        # Generate synthetic features (precipitation, temperature, etc.)
        np.random.seed(basin_id)  # Reproducible data
        
        # Precipitation (non-negative, seasonal pattern)
        precip = np.abs(np.random.normal(5, 2, time_steps))
        precip += 3 * np.sin(2 * np.pi * np.arange(time_steps) / 365)  # Seasonal
        
        # Temperature (seasonal pattern)
        temp = 15 + 10 * np.sin(2 * np.pi * np.arange(time_steps) / 365)
        temp += np.random.normal(0, 2, time_steps)
        
        # Soil moisture (autocorrelated)
        soil_moisture = np.zeros(time_steps)
        soil_moisture[0] = 0.5
        for t in range(1, time_steps):
            soil_moisture[t] = 0.8 * soil_moisture[t-1] + 0.1 * precip[t] + np.random.normal(0, 0.1)
        
        features = np.column_stack([precip, temp, soil_moisture])
        
        # Synthetic streamflow (function of features with noise)
        streamflow = (0.3 * precip + 
                     -0.1 * np.maximum(temp - 10, 0) +  # Evaporation effect
                     0.2 * soil_moisture + 
                     np.random.normal(0, 1, time_steps))
        streamflow = np.maximum(streamflow, 0.1)  # Minimum flow
        
        basin_data[f'basin_{basin_id:03d}'] = {
            'features': features,
            'targets': streamflow
        }
    
    return basin_data


def demonstrate_traditional_transfer():
    """
    Demonstration of traditional LSTM transfer learning
    """
    logger.info("=== Traditional LSTM Transfer Learning Demonstration ===")
    
    # Create synthetic data
    logger.info("Creating synthetic basin data...")
    source_basins = create_synthetic_basin_data(num_basins=3, time_steps=800)
    target_basin = create_synthetic_basin_data(num_basins=1, time_steps=400)['basin_000']
    
    # Initialize framework
    framework = TraditionalTransferFramework()
    
    # Step 1: Pre-train on source basins
    logger.info("\nStep 1: Pre-training on source basins...")
    pretrain_metrics = framework.pretrain_on_source(source_basins, epochs=30)
    logger.info(f"Pre-training metrics: {pretrain_metrics}")
    
    # Step 2: Transfer to target basin
    logger.info("\nStep 2: Transfer learning to target basin...")
    transfer_metrics = framework.transfer_to_target(
        target_basin, 
        fine_tune_epochs=15,
        freeze_layers=['lstm.weight_ih_l0']  # Freeze first layer input weights
    )
    logger.info(f"Transfer metrics: {transfer_metrics}")
    
    # Step 3: Evaluate performance
    logger.info("\nStep 3: Evaluating final performance...")
    test_metrics = framework.evaluate_on_test(target_basin)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Create comparison baseline (no transfer)
    logger.info("\nStep 4: Training baseline (no transfer) for comparison...")
    baseline_framework = TraditionalTransferFramework()
    baseline_data = {
        'features': target_basin['features'][:200],  # Limited data
        'targets': target_basin['targets'][:200]
    }
    baseline_X, baseline_y = baseline_framework.prepare_data(baseline_data)
    
    # Train baseline from scratch
    optimizer = torch.optim.AdamW(baseline_framework.model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    baseline_framework.model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = baseline_framework.model(baseline_X)
        loss = criterion(predictions, baseline_y)
        loss.backward()
        optimizer.step()
    
    baseline_metrics = baseline_framework.evaluate_on_test(target_basin)
    logger.info(f"Baseline metrics (no transfer): {baseline_metrics}")
    
    # Compare results
    logger.info("\n=== COMPARISON RESULTS ===")
    logger.info(f"Traditional Transfer - NSE: {test_metrics['NSE']:.4f}")
    logger.info(f"No Transfer Baseline - NSE: {baseline_metrics['NSE']:.4f}")
    logger.info(f"Transfer Improvement: {((test_metrics['NSE'] - baseline_metrics['NSE']) / baseline_metrics['NSE'] * 100):.1f}%")
    
    return {
        'transfer_performance': test_metrics,
        'baseline_performance': baseline_metrics,
        'improvement_pct': (test_metrics['NSE'] - baseline_metrics['NSE']) / baseline_metrics['NSE'] * 100
    }


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_traditional_transfer()
    
    logger.info("\n=== TRADITIONAL TRANSFER LSTM IMPLEMENTATION COMPLETE ===")
    logger.info("This module provides:")
    logger.info("1. Fair comparison baseline for foundation models")
    logger.info("2. Standardized hyperparameter protocols")
    logger.info("3. Traditional pre-train + fine-tune paradigm")
    logger.info("4. Comprehensive evaluation framework")

"""
Training script for Conditional U-Net CMAQ PM2.5 Emulator

This script trains a Conditional U-Net model to emulate CMAQ simulations
for PM2.5 concentration prediction based on emission control scenarios.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import custom modules
from src.model import get_unet_model
from src.dataloader import create_data_generator

# Constants
SEASON_NAMES = ['January', 'April', 'July', 'October']
EMISSION_SECTORS = ['ALL_POW', 'ALL_IND', 'ALL_MOB', 'ALL_RES', 'NH3_AGR', 'ALL_SLV', 'ALL_OTH']
N_REGIONS = 17
N_SECTORS = 7
GRID_HEIGHT = 82
GRID_WIDTH = 67

class PM25DataGenerator(tf.keras.utils.Sequence):
    """Data generator for batch training"""
    
    def __init__(self, X_ctrl, X_time, y, batch_size=32, shuffle=True):
        self.X_ctrl = X_ctrl
        self.X_time = X_time
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X_ctrl))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.X_ctrl) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_ctrl_batch = self.X_ctrl[batch_indices]
        X_time_batch = self.X_time[batch_indices]
        y_batch = self.y[batch_indices]
        return [X_ctrl_batch, X_time_batch], y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_training_data(data_path, season_index):
    """
    Load and prepare training data for a specific season
    
    Args:
        data_path: Path to dataset directory
        season_index: Season index (0-3)
    
    Returns:
        Tuple of (control_data, time_data, concentration_data)
    """
    # Season configurations
    season_configs = {
        0: {'slice': (0, 41*24), 'hours': 41*24},      # January
        1: {'slice': (41*24, 81*24), 'hours': 40*24},  # April  
        2: {'slice': (81*24, 122*24), 'hours': 41*24}, # July
        3: {'slice': (122*24, 163*24), 'hours': 41*24} # October
    }
    
    config = season_configs[season_index]
    
    # Load control matrix (119 scenarios)
    ctrl_matrix = pd.read_csv(f'{data_path}/control_matrix.csv', index_col=0)
    
    # Load PM2.5 concentration data
    conc_data = np.load(f'{data_path}/pm25_concentrations.npy')
    season_conc = conc_data[:, config['slice'][0]:config['slice'][1]]
    
    # Prepare control and time datasets
    n_scenarios = ctrl_matrix.shape[0]
    n_hours = config['hours']
    
    ctrl_expanded = []
    time_expanded = []
    
    for scenario_idx in range(n_scenarios):
        # Repeat control matrix for each hour
        scenario_ctrl = np.repeat(
            ctrl_matrix.values[scenario_idx:scenario_idx+1], 
            n_hours, 
            axis=0
        )
        ctrl_expanded.append(scenario_ctrl)
        
        # Create time indices
        time_indices = np.arange(n_hours, dtype=np.float32)
        time_expanded.append(time_indices)
    
    ctrl_dataset = np.vstack(ctrl_expanded)
    time_dataset = np.concatenate(time_expanded)
    conc_dataset = season_conc.reshape(-1, GRID_HEIGHT, GRID_WIDTH, 1)
    
    return ctrl_dataset, time_dataset, conc_dataset

def train_with_cross_validation(model_fn, X_ctrl, X_time, y, n_folds=5, epochs=300):
    """
    Train model using k-fold cross-validation
    
    Args:
        model_fn: Function to create model
        X_ctrl: Control matrix data
        X_time: Time indices
        y: Target PM2.5 concentrations
        n_folds: Number of cross-validation folds
        epochs: Training epochs per fold
    
    Returns:
        List of trained models and validation scores
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_ctrl)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")
        
        # Split data
        X_ctrl_train, X_ctrl_val = X_ctrl[train_idx], X_ctrl[val_idx]
        X_time_train, X_time_val = X_time[train_idx], X_time[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model = model_fn()
        
        # Create data generators
        train_gen = PM25DataGenerator(
            X_ctrl_train, X_time_train, y_train, 
            batch_size=256, shuffle=True
        )
        val_gen = PM25DataGenerator(
            X_ctrl_val, X_time_val, y_val,
            batch_size=256, shuffle=False
        )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss = model.evaluate(val_gen, verbose=0)
        scores.append(val_loss)
        models.append(model)
        
        print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"Average Loss: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return models, scores

def main():
    parser = argparse.ArgumentParser(
        description='Train Conditional U-Net for CMAQ PM2.5 Emulation'
    )
    parser.add_argument('--data_path', type=str, default='./datasets',
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='Path to save trained models')
    parser.add_argument('--season', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Season index (0: Jan, 1: Apr, 2: Jul, 3: Oct)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Initial learning rate')
    
    args = parser.parse_args()
    
    print(f"Training Conditional U-Net for {SEASON_NAMES[args.season]}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Cross-Validation Folds: {args.cv_folds}")
    print(f"  - Learning Rate: {args.learning_rate}")
    
    # Load data
    print("\nLoading training data...")
    X_ctrl, X_time, y = load_training_data(args.data_path, args.season)
    print(f"Data shape: Control={X_ctrl.shape}, Time={X_time.shape}, PM2.5={y.shape}")
    
    # Split into train/test
    print("Splitting data (60 train / 59 test scenarios)...")
    n_scenarios = 119
    scenario_indices = np.arange(n_scenarios)
    train_scenarios, test_scenarios = train_test_split(
        scenario_indices, test_size=59, random_state=42, shuffle=True
    )
    
    # Create model function
    def create_model():
        return get_unet_model(
            input_shape=(N_REGIONS * N_SECTORS,),
            output_shape=(GRID_HEIGHT, GRID_WIDTH, 1),
            time_dim=128,
            learning_rate=args.learning_rate
        )
    
    # Train with cross-validation
    if args.cv_folds > 1:
        print(f"\nStarting {args.cv_folds}-fold cross-validation training...")
        models, scores = train_with_cross_validation(
            create_model, X_ctrl, X_time, y,
            n_folds=args.cv_folds,
            epochs=args.epochs
        )
        
        # Save best model
        best_idx = np.argmin(scores)
        best_model = models[best_idx]
        print(f"\nBest model: Fold {best_idx + 1} with loss {scores[best_idx]:.4f}")
    else:
        print("\nTraining single model...")
        model = create_model()
        model.fit(
            [X_ctrl, X_time], y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
            ]
        )
        best_model = model
    
    # Save model
    os.makedirs(args.model_path, exist_ok=True)
    model_file = f"{args.model_path}/cond_unet_pm25_{SEASON_NAMES[args.season].lower()}.h5"
    best_model.save(model_file)
    print(f"\nModel saved to {model_file}")
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    test_loss = best_model.evaluate([X_ctrl, X_time], y, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")

if __name__ == "__main__":
    main()
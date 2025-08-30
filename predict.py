"""
PM2.5 Concentration Prediction using Conditional U-Net

This script loads a trained Conditional U-Net model and generates PM2.5 
concentration predictions for given emission scenarios.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import custom modules
from src.alloc import allocation
from src.model import get_unet_model

# Constants
SEASON_NAMES = ['January', 'April', 'July', 'October']
EMISSION_SECTORS = ['ALL_POW', 'ALL_IND', 'ALL_MOB', 'ALL_RES', 'NH3_AGR', 'ALL_SLV', 'ALL_OTH']
REGION_CODES = {
    'A': 'Seoul', 'B': 'Incheon', 'C': 'Busan', 'D': 'Daegu',
    'E': 'Gwangju', 'F': 'Gyeonggi', 'G': 'Gangwon', 'H': 'Chungbuk',
    'I': 'Chungnam', 'J': 'Gyeongbuk', 'K': 'Gyeongnam', 'L': 'Jeonbuk',
    'M': 'Jeonnam', 'N': 'Jeju', 'O': 'Daejeon', 'P': 'Ulsan', 'Q': 'Sejong'
}

def load_data(data_path, season_index, n_scenarios=119):
    """
    Load control matrix and PM2.5 concentration data
    
    Args:
        data_path: Path to dataset directory
        season_index: Index of season (0-3)
        n_scenarios: Number of emission scenarios
    
    Returns:
        Tuple of (control_data, time_data, concentration_data)
    """
    # Season-specific parameters
    season_slices = [
        slice(0, 41*24),      # January
        slice(41*24, 81*24),  # April
        slice(81*24, 122*24), # July
        slice(122*24, 163*24) # October
    ]
    time_lengths = [41*24, 40*24, 41*24, 41*24]
    
    # Load control matrix
    ctrl_matrix = pd.read_csv(f'{data_path}/control_matrix.csv', index_col=0)
    
    # Load PM2.5 concentration data
    conc_data = np.load(f'{data_path}/pm25_concentrations.npy')
    conc_data = conc_data[:, season_slices[season_index]]
    
    # Prepare datasets
    ctrl_dataset = []
    time_dataset = []
    time_steps = list(range(time_lengths[season_index]))
    
    for i in range(n_scenarios):
        time_dataset.append(time_steps.copy())
        ctrl_dataset.append([ctrl_matrix.values[i] for _ in range(len(time_steps))])
    
    ctrl_dataset = np.array(ctrl_dataset).squeeze()
    time_dataset = np.array(time_dataset).astype(np.float32)
    
    return ctrl_dataset, time_dataset, conc_data

def predict_pm25(model_path, test_data, season_index):
    """
    Generate PM2.5 predictions using trained model
    
    Args:
        model_path: Path to saved model
        test_data: Tuple of (control, time, concentration) test data
        season_index: Index of season for model selection
    
    Returns:
        Array of predicted PM2.5 concentrations
    """
    # Unpack test data
    X_ctrl, X_time, y_true = test_data
    
    # Reshape for model input
    X_ctrl_hour = X_ctrl.reshape(-1, 119)
    X_time_hour = X_time.reshape(-1)
    
    # Load trained model
    model = tf.keras.models.load_model(
        f'{model_path}/cond_unet_pm25_{SEASON_NAMES[season_index].lower()}'
    )
    
    # Generate predictions
    y_pred = model.predict([X_ctrl_hour, X_time_hour], batch_size=256)
    
    return y_pred

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics for predictions
    
    Args:
        y_true: Ground truth PM2.5 concentrations
        y_pred: Predicted PM2.5 concentrations
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Flatten arrays for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    # Mean Normalized Error
    mask = y_true_flat > 0
    mne = np.mean(np.abs(y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]) * 100
    
    # R-squared
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'MeanNE': mne,
        'R2': r2
    }

def main():
    parser = argparse.ArgumentParser(
        description='Generate PM2.5 predictions using Conditional U-Net'
    )
    parser.add_argument('--data_path', type=str, default='./datasets',
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='Path to saved models')
    parser.add_argument('--season', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Season index (0: Jan, 1: Apr, 2: Jul, 3: Oct)')
    parser.add_argument('--output_path', type=str, default='./results',
                        help='Path to save prediction results')
    parser.add_argument('--test_size', type=int, default=60,
                        help='Number of scenarios for testing')
    
    args = parser.parse_args()
    
    print(f"Loading data for {SEASON_NAMES[args.season]}...")
    ctrl_data, time_data, conc_data = load_data(
        args.data_path, args.season
    )
    
    # Split data into train/test
    print("Splitting data into train/test sets...")
    X_ctrl_train, X_ctrl_test, X_time_train, X_time_test, y_train, y_test = \
        train_test_split(
            ctrl_data, time_data, conc_data,
            test_size=args.test_size,
            random_state=42,
            shuffle=True
        )
    
    # Generate predictions
    print("Generating PM2.5 predictions...")
    y_pred = predict_pm25(
        args.model_path,
        (X_ctrl_test, X_time_test, y_test),
        args.season
    )
    
    # Reshape predictions
    time_length = y_test.shape[1]
    y_test_reshaped = y_test.reshape(-1, time_length, 82, 67, 1)
    y_pred_reshaped = y_pred.reshape(-1, time_length, 82, 67, 1)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    metrics = evaluate_predictions(y_test_reshaped, y_pred_reshaped)
    
    print(f"\nPrediction Results for {SEASON_NAMES[args.season]}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f} µg/m³")
    print(f"  Mean Normalized Error: {metrics['MeanNE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    output_file = f"{args.output_path}/predictions_{SEASON_NAMES[args.season].lower()}.npz"
    np.savez_compressed(
        output_file,
        predictions=y_pred_reshaped,
        ground_truth=y_test_reshaped,
        metrics=metrics
    )
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
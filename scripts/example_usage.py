"""
Example usage of the Conditional U-Net CMAQ PM2.5 Emulator

This script demonstrates how to use the trained model for 
various emission reduction scenarios.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the model
from src.model import get_unet_model

# Region and sector definitions
REGIONS = {
    'Seoul': 0, 'Incheon': 1, 'Busan': 2, 'Daegu': 3,
    'Gwangju': 4, 'Gyeonggi': 5, 'Gangwon': 6, 'Chungbuk': 7,
    'Chungnam': 8, 'Gyeongbuk': 9, 'Gyeongnam': 10, 'Jeonbuk': 11,
    'Jeonnam': 12, 'Jeju': 13, 'Daejeon': 14, 'Ulsan': 15, 'Sejong': 16
}

SECTORS = {
    'Power Plants': 0,
    'Industry': 1,
    'Mobile Sources': 2,
    'Residential': 3,
    'Agriculture': 4,
    'Solvents': 5,
    'Others': 6
}

class EmissionScenario:
    """Class to define and evaluate emission reduction scenarios"""
    
    def __init__(self):
        # Initialize with baseline (1.0 = 100% of baseline emissions)
        self.control_matrix = np.ones((17, 7), dtype=np.float32)
    
    def set_sector_reduction(self, sector, reduction_factor):
        """
        Set emission reduction for a specific sector
        
        Args:
            sector: Sector name (from SECTORS dict)
            reduction_factor: Emission factor (0.5 = 50% reduction)
        """
        if sector in SECTORS:
            sector_idx = SECTORS[sector]
            self.control_matrix[:, sector_idx] = reduction_factor
            print(f"Set {sector} emissions to {reduction_factor*100:.0f}% of baseline")
    
    def set_regional_reduction(self, region, sector, reduction_factor):
        """
        Set emission reduction for a specific region and sector
        
        Args:
            region: Region name (from REGIONS dict)
            sector: Sector name (from SECTORS dict)
            reduction_factor: Emission factor (0.5 = 50% reduction)
        """
        if region in REGIONS and sector in SECTORS:
            region_idx = REGIONS[region]
            sector_idx = SECTORS[sector]
            self.control_matrix[region_idx, sector_idx] = reduction_factor
            print(f"Set {region} {sector} emissions to {reduction_factor*100:.0f}% of baseline")
    
    def get_control_vector(self):
        """Convert control matrix to model input format"""
        return self.control_matrix.flatten()
    
    def evaluate(self, model, time_steps):
        """
        Evaluate PM2.5 concentrations for this scenario
        
        Args:
            model: Trained Conditional U-Net model
            time_steps: Array of time indices to evaluate
        
        Returns:
            Predicted PM2.5 concentrations
        """
        control_vector = self.get_control_vector()
        n_times = len(time_steps)
        
        # Prepare inputs
        X_ctrl = np.tile(control_vector, (n_times, 1))
        X_time = np.array(time_steps, dtype=np.float32)
        
        # Generate predictions
        predictions = model.predict([X_ctrl, X_time], batch_size=32)
        
        return predictions

def compare_scenarios(model, scenarios, time_steps, region_mask=None):
    """
    Compare multiple emission scenarios
    
    Args:
        model: Trained model
        scenarios: Dictionary of {name: EmissionScenario}
        time_steps: Time steps to evaluate
        region_mask: Optional mask for specific region analysis
    
    Returns:
        Dictionary of results
    """
    results = {}
    
    for name, scenario in scenarios.items():
        print(f"\nEvaluating scenario: {name}")
        predictions = scenario.evaluate(model, time_steps)
        
        if region_mask is not None:
            # Calculate average for specific region
            masked_pred = predictions * region_mask[np.newaxis, :, :, np.newaxis]
            avg_concentration = np.mean(masked_pred[masked_pred > 0])
        else:
            # Calculate domain average
            avg_concentration = np.mean(predictions)
        
        results[name] = {
            'predictions': predictions,
            'average': avg_concentration
        }
    
    return results

def visualize_scenario_impact(baseline, scenario, title="Emission Scenario Impact"):
    """
    Visualize the difference between baseline and scenario
    
    Args:
        baseline: Baseline PM2.5 predictions
        scenario: Scenario PM2.5 predictions  
        title: Plot title
    """
    # Calculate difference
    difference = scenario - baseline
    avg_difference = np.mean(difference, axis=0).squeeze()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Baseline
    im1 = axes[0].imshow(np.mean(baseline, axis=0).squeeze(), cmap='YlOrRd', vmin=0)
    axes[0].set_title('Baseline PM2.5')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='µg/m³')
    
    # Scenario
    im2 = axes[1].imshow(np.mean(scenario, axis=0).squeeze(), cmap='YlOrRd', vmin=0)
    axes[1].set_title('Scenario PM2.5')
    axes[1].set_xlabel('Longitude')
    plt.colorbar(im2, ax=axes[1], label='µg/m³')
    
    # Difference
    max_diff = np.abs(avg_difference).max()
    im3 = axes[2].imshow(avg_difference, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title('PM2.5 Change')
    axes[2].set_xlabel('Longitude')
    plt.colorbar(im3, ax=axes[2], label='µg/m³')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Average baseline PM2.5: {np.mean(baseline):.2f} µg/m³")
    print(f"Average scenario PM2.5: {np.mean(scenario):.2f} µg/m³")
    print(f"Average reduction: {-np.mean(difference):.2f} µg/m³ ({-np.mean(difference)/np.mean(baseline)*100:.1f}%)")

# Example usage
if __name__ == "__main__":
    print("Conditional U-Net CMAQ PM2.5 Emulator - Example Usage")
    print("=" * 50)
    
    # Load model (replace with actual model path)
    print("\nLoading trained model...")
    # model = tf.keras.models.load_model('models/cond_unet_pm25_january.h5')
    
    # Define scenarios
    print("\nDefining emission scenarios...")
    
    # Baseline scenario (no reduction)
    baseline = EmissionScenario()
    
    # Scenario 1: 30% reduction in power plants nationwide
    scenario1 = EmissionScenario()
    scenario1.set_sector_reduction('Power Plants', 0.7)
    
    # Scenario 2: 50% reduction in mobile sources in Seoul metropolitan area
    scenario2 = EmissionScenario()
    scenario2.set_regional_reduction('Seoul', 'Mobile Sources', 0.5)
    scenario2.set_regional_reduction('Gyeonggi', 'Mobile Sources', 0.5)
    scenario2.set_regional_reduction('Incheon', 'Mobile Sources', 0.5)
    
    # Scenario 3: Combined reduction strategy
    scenario3 = EmissionScenario()
    scenario3.set_sector_reduction('Power Plants', 0.8)
    scenario3.set_sector_reduction('Industry', 0.85)
    scenario3.set_sector_reduction('Mobile Sources', 0.7)
    
    # Package scenarios
    scenarios = {
        'Baseline': baseline,
        '30% Power Plant Reduction': scenario1,
        '50% Mobile Reduction (Seoul Metro)': scenario2,
        'Combined Strategy': scenario3
    }
    
    # Evaluate for first 24 hours
    time_steps = np.arange(24)
    
    print("\nEvaluating scenarios...")
    # results = compare_scenarios(model, scenarios, time_steps)
    
    print("\nScenario comparison complete!")
    print("\nExample demonstrates:")
    print("- Creating emission reduction scenarios")
    print("- Evaluating PM2.5 impacts")
    print("- Comparing multiple strategies")
    print("- Visualizing spatial impacts")
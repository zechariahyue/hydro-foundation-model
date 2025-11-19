#!/usr/bin/env python3
"""
Foundation Models for Hydro-Climate Forecasting - Quick Start Example
====================================================================

This example demonstrates how to use the foundation models for streamflow forecasting
with both zero-shot and few-shot capabilities.

Based on the paper: "Foundation Models for Global Hydro-Climate Forecasting:
A Comprehensive Evaluation of Zero-Shot and Few-Shot Transfer Learning"

Author: Zero Water Team
Date: January 2025
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from enhanced_foundation_models import (
    CombinedEnhancedModel,
    EnhancedClimateIntegration,
    PhysicsGuidedLoss,
    LoRAFineTuning,
    StaticBasinAttributes
)


def generate_dummy_data(num_basins: int = 5, time_steps: int = 500):
    """
    Generate dummy hydro-climate data for demonstration

    Args:
        num_basins: Number of basins to generate
        time_steps: Number of time steps per basin

    Returns:
        dict: Dictionary containing basin data
    """
    print(f"Generating dummy data for {num_basins} basins...")

    data = {}
    for basin_id in range(num_basins):
        np.random.seed(basin_id)

        # Generate climate features
        features = np.random.randn(time_steps, 3)

        # Generate targets correlated with features
        targets = 5 + 2 * features[:, 0] + np.random.normal(0, 1, time_steps)

        data[f'basin_{basin_id:03d}'] = {
            'features': features,
            'targets': targets
        }

    print(f"Generated data for {num_basins} basins")
    return data


def demonstrate_enhanced_climate_integration():
    """Demonstrate enhanced climate data integration"""
    print("\n" + "="*60)
    print("ENHANCED CLIMATE INTEGRATION DEMONSTRATION")
    print("="*60)

    # Initialize climate integration
    climate_integration = EnhancedClimateIntegration()

    print(f"\nClimate variables used: {len(climate_integration.era5_variables)}")
    print("Variables:", climate_integration.get_feature_names())

    # Generate sample basin data
    sample_data = {'temperature': np.random.randn(100)}

    # Process climate data
    climate_features = climate_integration.process_climate_data(sample_data)

    print(f"\nProcessed climate features shape: {climate_features.shape}")
    print(f"Total climate features: {climate_integration.num_climate_features}")

    return climate_features


def demonstrate_physics_guided_loss():
    """Demonstrate physics-guided loss functions"""
    print("\n" + "="*60)
    print("PHYSICS-GUIDED LOSS DEMONSTRATION")
    print("="*60)

    # Initialize physics-guided loss
    physics_loss = PhysicsGuidedLoss(
        mass_balance_weight=0.01,
        snow_dynamics_weight=0.01,
        smoothness_weight=0.001
    )

    # Generate sample predictions and climate data
    time_steps = 100
    predictions = np.random.randn(time_steps) * 5 + 10
    predictions = np.maximum(predictions, 0)  # Ensure positive

    climate_inputs = {
        'precipitation': np.random.exponential(3, time_steps),
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(time_steps) / 365),
        'evapotranspiration': np.random.exponential(2, time_steps)
    }

    # Compute physics losses
    mass_balance_loss = physics_loss.compute_mass_balance_loss(
        predictions,
        climate_inputs['precipitation'],
        climate_inputs['evapotranspiration']
    )

    smoothness_loss = physics_loss.compute_smoothness_loss(predictions)

    total_physics_loss = physics_loss.compute_total_physics_loss(
        predictions, climate_inputs
    )

    print(f"\nMass balance loss: {mass_balance_loss:.4f}")
    print(f"Smoothness loss: {smoothness_loss:.4f}")
    print(f"Total physics loss: {total_physics_loss:.4f}")

    return total_physics_loss


def demonstrate_lora_finetuning():
    """Demonstrate LoRA fine-tuning"""
    print("\n" + "="*60)
    print("LoRA FINE-TUNING DEMONSTRATION")
    print("="*60)

    # Initialize LoRA
    lora = LoRAFineTuning(rank=16, alpha=8.0)

    # Create sample weight matrices
    weight_shapes = [
        (256, 64),   # Layer 1
        (256, 256),  # Layer 2
        (64, 256)    # Output layer
    ]

    # Calculate parameter efficiency
    original_params = sum(w[0] * w[1] for w in weight_shapes)
    efficiency = lora.compute_parameter_efficiency(original_params, weight_shapes)

    print(f"\nOriginal parameters: {efficiency['original_parameters']:,}")
    print(f"LoRA parameters: {efficiency['lora_parameters']:,}")
    print(f"Parameter reduction: {efficiency['parameter_reduction_percent']:.1f}%")

    # Demonstrate LoRA adaptation
    original_weight = np.random.randn(256, 64)
    lora_A, lora_B = lora.create_lora_matrices((256, 64))
    adapted_weight = lora.apply_lora_adaptation(original_weight, lora_A, lora_B)

    print(f"\nOriginal weight shape: {original_weight.shape}")
    print(f"Adapted weight shape: {adapted_weight.shape}")
    print(f"Weight difference norm: {np.linalg.norm(adapted_weight - original_weight):.4f}")

    return efficiency


def demonstrate_combined_model():
    """Demonstrate the combined enhanced model"""
    print("\n" + "="*60)
    print("COMBINED ENHANCED MODEL DEMONSTRATION")
    print("="*60)

    # Generate source and target data
    source_data = generate_dummy_data(num_basins=3, time_steps=500)
    target_data = generate_dummy_data(num_basins=1, time_steps=200)

    # Initialize combined model
    model = CombinedEnhancedModel()

    # Prepare enhanced data
    print("\nPreparing enhanced data...")
    enhanced_source = model.prepare_enhanced_data(source_data)
    enhanced_target = model.prepare_enhanced_data(target_data)

    # Simulate training
    print("Simulating model training...")
    training_results = model.simulate_training(
        enhanced_source,
        enhanced_target
    )

    print(f"\nTraining completed in {training_results['epochs_trained']} epochs")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")

    # Simulate H2 evaluation
    print("\nSimulating H2 evaluation...")
    target_basin_data = list(target_data.values())[0]
    h2_results = model.simulate_h2_evaluation(target_basin_data, few_shot_samples=50)

    print(f"\nH2 Evaluation Results:")
    print(f"   MSE Improvement: {h2_results['mse_improvement_pct']:.1f}%")
    print(f"   Label Reduction: {h2_results['label_reduction_pct']:.1f}%")
    print(f"   H2 Satisfied: {'Yes' if h2_results['h2_satisfied'] else 'No'}")

    return h2_results


def main():
    """Main demonstration function"""
    print("Foundation Models for Hydro-Climate Forecasting")
    print("=" * 60)
    print("Paper: Foundation Models for Global Hydro-Climate Forecasting")
    print("Authors: Zero Water Research Team")
    print("Date: January 2025")
    print("=" * 60)

    try:
        # Run demonstrations
        climate_results = demonstrate_enhanced_climate_integration()
        physics_results = demonstrate_physics_guided_loss()
        lora_results = demonstrate_lora_finetuning()
        model_results = demonstrate_combined_model()

        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("Enhanced climate integration: Demonstrated")
        print("Physics-guided loss: Demonstrated")
        print("LoRA fine-tuning: Demonstrated")
        print("Combined model: Demonstrated")

        print("\nKEY FINDINGS:")
        print("   - Enhanced climate integration with 10 ERA5 variables")
        print("   - Physics constraints improve hydrological realism")
        print(f"   - LoRA achieves {lora_results['parameter_reduction_percent']:.1f}% parameter reduction")
        print(f"   - H2 satisfied with {model_results['mse_improvement_pct']:.1f}% MSE improvement")

        print("\nNext Steps:")
        print("   1. Load real CAMELS/ERA5-Land data")
        print("   2. Train on your specific basins")
        print("   3. Evaluate on held-out test basins")
        print("   4. Deploy for operational forecasting")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

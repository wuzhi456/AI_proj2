"""
Main script for Adult Census Income Prediction
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from model.neural_network import create_model
from train.data_preprocessing import DataPreprocessor
from train.trainer import Trainer


def main():
    """Main function to train model and generate predictions."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # File paths
    project_root = os.path.join(os.path.dirname(__file__), '..')
    train_data_path = os.path.join(project_root, 'traindata.csv')
    train_label_path = os.path.join(project_root, 'trainlabel.txt')
    test_data_path = os.path.join(project_root, 'testdata.csv')
    output_path = os.path.join(project_root, 'testlabel.txt')
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    train_df, train_labels, test_df = preprocessor.load_data(
        train_data_path, train_label_path, test_data_path
    )
    
    X_train_scaled, y_train, X_test_scaled, feature_names = preprocessor.preprocess(
        train_df, train_labels, test_df
    )
    
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    
    # Split training data into train and validation sets
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]}")
    print(f"Validation set: {X_val.shape[0]}")
    
    # Create model
    print("\n2. Creating neural network model...")
    input_dim = X_train.shape[1]
    model = create_model(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3
    )
    
    print(f"Model architecture:")
    print(model)
    
    # Train model
    print("\n3. Training model...")
    trainer = Trainer(model, device=device, learning_rate=0.001)
    history = trainer.fit(
        X_train, y_train_split,
        X_val, y_val,
        epochs=50,
        batch_size=128,
        verbose=True
    )
    
    # Evaluate on validation set
    print("\n4. Evaluating model...")
    metrics, val_predictions = trainer.evaluate(X_val, y_val)
    
    print("\nValidation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Make predictions on test set
    print("\n5. Making predictions on test set...")
    test_predictions = trainer.predict(X_test_scaled)
    
    # Save predictions
    print(f"\n6. Saving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        for pred in test_predictions:
            f.write(f"{pred}\n")
    
    print("\nPrediction distribution:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} ({count/len(test_predictions)*100:.2f}%)")
    
    print("\n✓ Training and prediction completed successfully!")
    print(f"✓ Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

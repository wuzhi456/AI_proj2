"""
Main script for Adult Census Income Prediction with Interactive Visualization
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.neural_network import create_model
from train.data_preprocessing import DataPreprocessor
from train.trainer import Trainer

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def visualize_data_distribution(train_labels):
    """Visualize the distribution of training labels."""
    unique, counts = np.unique(train_labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    
    plt.figure(figsize=(8, 6))
    plt.bar(['<=50K (0)', '>50K (1)'], [label_dist[0], label_dist[1]], 
            color=['#3498db', '#e74c3c'])
    plt.ylabel('Count', fontsize=12)
    plt.title('Training Data Label Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, (label, count) in enumerate(label_dist.items()):
        plt.text(i, count, f'{count}\n({count/sum(counts)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nClass distribution:")
    print(f"  <=50K (0): {label_dist[0]} ({label_dist[0]/sum(counts)*100:.2f}%)")
    print(f"  >50K (1): {label_dist[1]} ({label_dist[1]/sum(counts)*100:.2f}%)")


def main():
    """Main function to train model and generate predictions with visualization."""
    
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
    print("\n" + "="*60)
    print("1. Loading and Exploring Data")
    print("="*60)
    preprocessor = DataPreprocessor()
    train_df, train_labels, test_df = preprocessor.load_data(
        train_data_path, train_label_path, test_data_path
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Number of features: {len(train_df.columns)}")
    
    # Visualize label distribution
    visualize_data_distribution(train_labels)
    
    # Preprocess data
    print("\n" + "="*60)
    print("2. Preprocessing Data")
    print("="*60)
    X_train_scaled, y_train, X_test_scaled, feature_names = preprocessor.preprocess(
        train_df, train_labels, test_df
    )
    
    print(f"  Features after preprocessing: {X_train_scaled.shape[1]}")
    
    # Split training data into train and validation sets
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    
    # Create model
    print("\n" + "="*60)
    print("3. Building Neural Network Model")
    print("="*60)
    input_dim = X_train.shape[1]
    model = create_model(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model with visualization
    print("\n" + "="*60)
    print("4. Training Model")
    print("="*60)
    trainer = Trainer(model, device=device, learning_rate=0.001)
    history = trainer.fit(
        X_train, y_train_split,
        X_val, y_val,
        epochs=50,
        batch_size=128,
        verbose=True
    )
    
    # Visualize training history
    print("\n" + "="*60)
    print("5. Training Results Visualization")
    print("="*60)
    trainer.plot_training_history()
    plt.show()
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("6. Model Evaluation")
    print("="*60)
    metrics, val_predictions = trainer.evaluate(X_val, y_val)
    
    print("\nValidation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize():20s}: {value:.4f}")
    
    # Visualize confusion matrix
    trainer.plot_confusion_matrix(y_val, val_predictions)
    plt.show()
    
    # Display classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_val, val_predictions, 
                              target_names=['<=50K', '>50K'],
                              digits=4))
    
    # Make predictions on test set
    print("\n" + "="*60)
    print("7. Generating Test Predictions")
    print("="*60)
    test_predictions = trainer.predict(X_test_scaled)
    
    # Visualize test predictions distribution
    unique_test, counts_test = np.unique(test_predictions, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['<=50K (0)', '>50K (1)'], counts_test, color=['#3498db', '#e74c3c'])
    plt.ylabel('Count', fontsize=12)
    plt.title('Test Set Predictions Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, count in enumerate(counts_test):
        plt.text(i, count, f'{count}\n({count/len(test_predictions)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction distribution:")
    for label, count in zip(unique_test, counts_test):
        class_name = '<=50K' if label == 0 else '>50K'
        print(f"  {class_name} ({label}): {count} ({count/len(test_predictions)*100:.2f}%)")
    
    # Save predictions
    print("\n" + "="*60)
    print("8. Saving Predictions")
    print("="*60)
    with open(output_path, 'w') as f:
        for pred in test_predictions:
            f.write(f"{pred}\n")
    
    print(f"✓ Predictions saved to: {output_path}")
    print(f"✓ Total predictions: {len(test_predictions)}")
    
    # Final summary
    print("\n" + "="*60)
    print("Training and Prediction Completed Successfully!")
    print("="*60)
    print(f"\nPerformance Summary:")
    print(f"  Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nOutputs:")
    print(f"  ✓ Test predictions: {output_path}")
    print(f"  ✓ All visualizations displayed")


if __name__ == "__main__":
    main()

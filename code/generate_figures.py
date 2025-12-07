"""
Script to generate figures for the report
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
from sklearn.metrics import classification_report, roc_curve, auc

from model.neural_network import create_model
from train.data_preprocessing import DataPreprocessor
from train.trainer import Trainer

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Create figures directory
os.makedirs('../figures', exist_ok=True)

print("Generating figures for the report...")

# File paths
project_root = os.path.join(os.path.dirname(__file__), '..')
train_data_path = os.path.join(project_root, 'traindata.csv')
train_label_path = os.path.join(project_root, 'trainlabel.txt')
test_data_path = os.path.join(project_root, 'testdata.csv')

# Load data
print("\n1. Loading data...")
preprocessor = DataPreprocessor()
train_df, train_labels, test_df = preprocessor.load_data(
    train_data_path, train_label_path, test_data_path
)

# Label distribution
print("\n2. Creating label distribution figure...")
unique, counts = np.unique(train_labels, return_counts=True)
label_dist = dict(zip(unique, counts))

plt.figure(figsize=(8, 6))
plt.bar(['<=50K (0)', '>50K (1)'], [label_dist[0], label_dist[1]], color=['#3498db', '#e74c3c'])
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Income Labels in Training Data', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(label_dist.items()):
    plt.text(i, count, f'{count}\n({count/sum(counts)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/label_distribution.png")

# Preprocess data
print("\n3. Preprocessing data...")
X_train_scaled, y_train, X_test_scaled, feature_names = preprocessor.preprocess(
    train_df, train_labels, test_df
)

# Split
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Create and train model
print("\n4. Training model...")
device = 'cpu'
input_dim = X_train.shape[1]
model = create_model(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3)
trainer = Trainer(model, device=device, learning_rate=0.001)

history = trainer.fit(
    X_train, y_train_split,
    X_val, y_val,
    epochs=50,
    batch_size=128,
    verbose=False
)

# Training history
print("\n5. Creating training history figure...")
trainer.plot_training_history(save_path='../figures/training_history.png')
plt.close()
print("✓ Saved: figures/training_history.png")

# Evaluate
print("\n6. Evaluating model...")
metrics, val_predictions = trainer.evaluate(X_val, y_val)

# Confusion matrix
print("\n7. Creating confusion matrix...")
trainer.plot_confusion_matrix(y_val, val_predictions, save_path='../figures/confusion_matrix.png')
plt.close()
print("✓ Saved: figures/confusion_matrix.png")

# ROC curve
print("\n8. Creating ROC curve...")
val_probabilities = trainer.predict_proba(X_val)
fpr, tpr, thresholds = roc_curve(y_val, val_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/roc_curve.png")

# Test predictions
print("\n9. Creating test predictions distribution...")
test_predictions = trainer.predict(X_test_scaled)
unique_test, counts_test = np.unique(test_predictions, return_counts=True)

plt.figure(figsize=(8, 6))
plt.bar(['<=50K (0)', '>50K (1)'], counts_test, color=['#3498db', '#e74c3c'])
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Predictions on Test Data', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, count in enumerate(counts_test):
    plt.text(i, count, f'{count}\n({count/len(test_predictions)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/test_predictions_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/test_predictions_distribution.png")

# Save performance metrics
print("\n10. Saving performance metrics...")
summary_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [
        f"{metrics['accuracy']:.4f}",
        f"{metrics['precision']:.4f}",
        f"{metrics['recall']:.4f}",
        f"{metrics['f1_score']:.4f}",
        f"{metrics['roc_auc']:.4f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../figures/performance_metrics.csv', index=False)
print("✓ Saved: figures/performance_metrics.csv")

# Model configuration
config_data = {
    'Parameter': ['Input Dimensions', 'Hidden Layers', 'Dropout Rate', 
                  'Learning Rate', 'Batch Size', 'Epochs', 'Optimizer'],
    'Value': [f"{input_dim}", "[128, 64, 32]", "0.3", "0.001", "128", "50", "Adam"]
}
config_df = pd.DataFrame(config_data)
config_df.to_csv('../figures/model_configuration.csv', index=False)
print("✓ Saved: figures/model_configuration.csv")

# Print metrics for report
print("\n" + "="*50)
print("VALIDATION METRICS FOR REPORT:")
print("="*50)
for metric_name, value in metrics.items():
    print(f"{metric_name.upper():20s}: {value:.4f}")
print("="*50)

print("\n✓ All figures generated successfully!")
print(f"✓ Figures saved in: {os.path.join(project_root, 'figures')}")

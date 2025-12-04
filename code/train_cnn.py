"""
Adult Census Income Prediction using CNN

This script implements a 1D Convolutional Neural Network (CNN) to predict
whether an individual's income exceeds $50K/year based on census data.

Author: AI Project 2
Date: December 2024
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class AdultCensusDataProcessor:
    """
    Data preprocessor for the Adult Census Income dataset.
    Handles missing values, encodes categorical features, and scales numerical features.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = [
            'workclass', 'education', 'marital.status', 'occupation',
            'relationship', 'race', 'sex', 'native.country'
        ]
        self.numerical_columns = [
            'age', 'fnlwgt', 'education.num', 'capital.gain',
            'capital.loss', 'hours.per.week'
        ]
        
    def load_data(self, train_data_path, train_label_path, test_data_path):
        """Load training and test data from CSV files."""
        # Load training data
        self.train_df = pd.read_csv(train_data_path)
        
        # Load training labels
        with open(train_label_path, 'r') as f:
            self.train_labels = np.array([int(line.strip()) for line in f.readlines()])
        
        # Load test data
        self.test_df = pd.read_csv(test_data_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Training labels shape: {self.train_labels.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        return self.train_df, self.train_labels, self.test_df
    
    def handle_missing_values(self, df):
        """Replace '?' with NaN and fill missing values."""
        df = df.replace('?', np.nan)
        
        # Fill missing values for categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        # Fill missing values for numerical columns with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        return df
    
    def encode_categorical(self, train_df, test_df):
        """Encode categorical features using LabelEncoder."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        for col in self.categorical_columns:
            if col in train_encoded.columns:
                # Fit encoder on combined data to handle unseen categories
                combined = pd.concat([train_encoded[col], test_encoded[col]], axis=0)
                le = LabelEncoder()
                le.fit(combined.astype(str))
                
                train_encoded[col] = le.transform(train_encoded[col].astype(str))
                test_encoded[col] = le.transform(test_encoded[col].astype(str))
                
                self.label_encoders[col] = le
        
        return train_encoded, test_encoded
    
    def scale_features(self, train_df, test_df):
        """Scale all features using StandardScaler."""
        # Get all feature columns
        feature_cols = [col for col in train_df.columns if col in 
                       self.categorical_columns + self.numerical_columns]
        
        # Fit scaler on training data
        self.scaler.fit(train_df[feature_cols])
        
        # Transform both training and test data
        train_scaled = self.scaler.transform(train_df[feature_cols])
        test_scaled = self.scaler.transform(test_df[feature_cols])
        
        return train_scaled, test_scaled
    
    def preprocess(self, train_data_path, train_label_path, test_data_path):
        """Complete preprocessing pipeline."""
        # Load data
        train_df, train_labels, test_df = self.load_data(
            train_data_path, train_label_path, test_data_path
        )
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        # Encode categorical features
        train_df, test_df = self.encode_categorical(train_df, test_df)
        
        # Scale features
        X_train, X_test = self.scale_features(train_df, test_df)
        
        print(f"\nPreprocessed training data shape: {X_train.shape}")
        print(f"Preprocessed test data shape: {X_test.shape}")
        
        return X_train, train_labels, X_test


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for tabular data classification.
    
    Architecture:
    - Reshape input to (batch, 1, features) for 1D convolution
    - Two 1D convolutional layers with batch normalization and dropout
    - Global average pooling
    - Fully connected layers with dropout
    - Sigmoid output for binary classification
    """
    
    def __init__(self, input_dim, dropout_rate=0.3):
        super(CNN1D, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = self.gap(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x


class CNNTrainer:
    """
    Trainer class for the CNN model.
    Handles training, validation, and prediction.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        return total_loss / total, correct / total
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        return total_loss / total, correct / total
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def predict(self, test_loader):
        """Generate predictions on test data."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predicted = (outputs > 0.5).long().squeeze()
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)


def main():
    """Main function to run the complete pipeline."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_path = os.path.join(base_path, 'traindata.csv')
    train_label_path = os.path.join(base_path, 'trainlabel.txt')
    test_data_path = os.path.join(base_path, 'testdata.csv')
    output_path = os.path.join(base_path, 'testlabel.txt')
    
    # Preprocess data
    print("=" * 60)
    print("Step 1: Data Preprocessing")
    print("=" * 60)
    processor = AdultCensusDataProcessor()
    X_train, y_train, X_test = processor.preprocess(
        train_data_path, train_label_path, test_data_path
    )
    
    # Split training data for validation
    print("\n" + "=" * 60)
    print("Step 2: Splitting Data for Validation")
    print("=" * 60)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    print(f"Training set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create data loaders
    batch_size = 64
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.LongTensor(y_train_split)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("\n" + "=" * 60)
    print("Step 3: Building CNN Model")
    print("=" * 60)
    input_dim = X_train.shape[1]
    model = CNN1D(input_dim=input_dim, dropout_rate=0.3)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Step 4: Training CNN Model")
    print("=" * 60)
    trainer = CNNTrainer(model, device=device, learning_rate=0.001)
    history = trainer.train(
        train_loader, val_loader,
        epochs=100, early_stopping_patience=15
    )
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Step 5: Final Evaluation on Validation Set")
    print("=" * 60)
    
    # Get validation predictions
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).long().squeeze()
            val_preds.extend(predicted.cpu().numpy())
    
    val_preds = np.array(val_preds)
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=['<=50K', '>50K']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_preds))
    
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    # Generate predictions on test set
    print("\n" + "=" * 60)
    print("Step 6: Generating Predictions on Test Set")
    print("=" * 60)
    test_predictions = trainer.predict(test_loader)
    print(f"Generated {len(test_predictions)} predictions")
    print(f"Class distribution: 0 (<=50K): {(test_predictions == 0).sum()}, 1 (>50K): {(test_predictions == 1).sum()}")
    
    # Save predictions
    with open(output_path, 'w') as f:
        for pred in test_predictions:
            f.write(f"{pred}\n")
    print(f"\nPredictions saved to: {output_path}")
    
    # Save model
    model_path = os.path.join(base_path, 'code', 'cnn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'val_accuracy': val_accuracy
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return val_accuracy


if __name__ == "__main__":
    main()

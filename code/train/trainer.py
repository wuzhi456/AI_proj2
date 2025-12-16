"""
Training utilities for Neural Network model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """
    Handles the training and evaluation of neural network models.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Neural network model
            device (str): Device to train on ('cpu' or 'cuda')
            learning_rate (float): Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # Changed from BCELoss for softmax
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).long()  # Convert to long for CrossEntropyLoss
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for accuracy calculation
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(targets, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).long()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(targets, predictions)
        
        return avg_loss, accuracy
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, verbose=True):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return self.history
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions.astype(int)
    
    def predict_proba(self, X_test):
        """
        Get probability predictions.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Probability predictions for class 1
        """
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_test_tensor)
            probabilities = outputs[:, 1].cpu().numpy()  # Get probability for class 1
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics, predictions
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['<=50K', '>50K'],
                    yticklabels=['<=50K', '>50K'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

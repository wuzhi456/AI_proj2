"""
Neural Network Model for Adult Census Income Classification
"""
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A multi-layer perceptron (MLP) for binary classification.
    
    Architecture:
    - Input layer
    - Hidden layers with ReLU activation and dropout
    - Output layer with sigmoid activation
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability for regularization
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.model(x)


def create_model(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
    """
    Factory function to create a neural network model.
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout probability
        
    Returns:
        NeuralNetwork: Initialized neural network model
    """
    return NeuralNetwork(input_dim, hidden_dims, dropout_rate)

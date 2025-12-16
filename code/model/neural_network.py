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
    - Hidden layers with ReLU activation
    - Output layer with softmax activation
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (2 classes for softmax)
        layers.append(nn.Linear(prev_dim, 2))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions (softmax probabilities)
        """
        return self.model(x)


def create_model(input_dim, hidden_dims=[128, 64, 32]):
    """
    Factory function to create a neural network model.
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (list): List of hidden layer dimensions
        
    Returns:
        NeuralNetwork: Initialized neural network model
    """
    return NeuralNetwork(input_dim, hidden_dims)

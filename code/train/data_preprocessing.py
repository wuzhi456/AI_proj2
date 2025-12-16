"""
Data preprocessing utilities for Adult Census Income dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for the Adult Census dataset.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, train_data_path, train_label_path, test_data_path):
        """
        Load training and test datasets.
        
        Args:
            train_data_path (str): Path to training data CSV
            train_label_path (str): Path to training labels TXT
            test_data_path (str): Path to test data CSV
            
        Returns:
            tuple: (train_df, train_labels, test_df)
        """
        # Load training data
        train_df = pd.read_csv(train_data_path)
        
        # Load training labels
        with open(train_label_path, 'r') as f:
            train_labels = np.array([int(line.strip()) for line in f.readlines()])
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        return train_df, train_labels, test_df
    
    def identify_feature_types(self, df):
        """
        Identify categorical and numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df = df.copy()
        
        # Replace ' ?' with NaN for easier handling
        df = df.replace(' ?', np.nan)
        
        # Fill missing categorical values with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
        
        # Fill missing numerical values with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
                
            if is_training:
                # Fit and transform for training data
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform for test data
                if col in self.label_encoders:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_labels else self.label_encoders[col].classes_[0])
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df, is_training=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
            
        Returns:
            np.ndarray: Scaled feature array
        """
        if is_training:
            scaled = self.scaler.fit_transform(df)
        else:
            scaled = self.scaler.transform(df)
        
        return scaled
    
    def downsample(self, X, y, random_state=42):
        """
        Downsample the majority class to balance the dataset.
        
        This method reduces the number of samples in the majority class
        to match the number of samples in the minority class.
        
        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Label array
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_downsampled, y_downsampled)
        """
        np.random.seed(random_state)
        
        # Find minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        minority_count = counts.min()
        
        # Get indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Randomly sample from majority class
        downsampled_majority_indices = np.random.choice(
            majority_indices, 
            size=minority_count, 
            replace=False
        )
        
        # Combine indices
        downsampled_indices = np.concatenate([minority_indices, downsampled_majority_indices])
        
        # Shuffle the combined indices
        np.random.shuffle(downsampled_indices)
        
        return X[downsampled_indices], y[downsampled_indices]
    
    def preprocess(self, train_df, train_labels, test_df, apply_downsampling=False, random_state=42):
        """
        Full preprocessing pipeline.
        
        Args:
            train_df (pd.DataFrame): Training data
            train_labels (np.ndarray): Training labels
            test_df (pd.DataFrame): Test data
            apply_downsampling (bool): Whether to apply downsampling to balance classes
            random_state (int): Random seed for downsampling reproducibility
            
        Returns:
            tuple: (X_train_scaled, y_train, X_test_scaled, feature_names)
            
        Note:
            Downsampling is applied after feature scaling. This ensures that
            the scaler is fitted on the full training data for more robust
            scaling parameters, which are then used consistently for both
            training and test data transformation.
        """
        # Identify feature types
        self.identify_feature_types(train_df)
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        # Encode categorical features
        train_df = self.encode_categorical_features(train_df, is_training=True)
        test_df = self.encode_categorical_features(test_df, is_training=False)
        
        # Store feature names
        self.feature_names = train_df.columns.tolist()
        
        # Scale features (fit on full training data for robust statistics)
        X_train_scaled = self.scale_features(train_df, is_training=True)
        X_test_scaled = self.scale_features(test_df, is_training=False)
        
        # Apply downsampling if requested (after scaling to preserve scaler statistics)
        if apply_downsampling:
            X_train_scaled, train_labels = self.downsample(X_train_scaled, train_labels, random_state)
        
        return X_train_scaled, train_labels, X_test_scaled, self.feature_names

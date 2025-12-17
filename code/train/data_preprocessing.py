"""
Data preprocessing utilities for Adult Census Income dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for the Adult Census dataset.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.onehot_encoder = None
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
        
        # Replace '?' with NaN for easier handling
        df = df.replace('?', np.nan)
        
        # Fill missing categorical values with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
        
        # Fill missing numerical values with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
            
        Returns:
            pd.DataFrame: Dataframe with one-hot encoded categorical features
        """
        df = df.copy()
        
        if not self.categorical_features:
            return df
        
        # Filter to only include categorical features that exist in the dataframe
        cat_cols = [col for col in self.categorical_features if col in df.columns]
        
        if not cat_cols:
            return df
        
        # Get categorical columns data
        cat_data = df[cat_cols].astype(str)
        
        if is_training:
            # Fit and transform for training data
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.onehot_encoder.fit_transform(cat_data)
        else:
            # Transform for test data
            if self.onehot_encoder is None:
                raise ValueError("One-hot encoder not fitted. Call encode_categorical_features with is_training=True first.")
            encoded = self.onehot_encoder.transform(cat_data)
        
        # Create column names for one-hot encoded features
        encoded_columns = self.onehot_encoder.get_feature_names_out(cat_cols)
        
        # Create dataframe with encoded features
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        
        # Drop original categorical columns and add encoded columns
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)
        
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

# Adult Census Income Prediction using CNN

This project implements a 1D Convolutional Neural Network (CNN) to predict whether an individual's income exceeds $50K/year based on census data from the 1994 Census bureau database.

## Project Structure

```
AI_proj2/
├── code/
│   ├── train_cnn.py      # Main CNN training script
│   ├── README.md         # This file
│   └── cnn_model.pth     # Saved model (generated after training)
├── traindata.csv         # Training data
├── trainlabel.txt        # Training labels
├── testdata.csv          # Test data
├── testlabel.txt         # Predicted labels (generated after training)
└── report.pdf            # Project report
```

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn torch
```

Or install all dependencies at once:

```bash
pip install numpy>=1.20.0 pandas>=1.3.0 scikit-learn>=1.0.0 torch>=1.10.0
```

## How to Run

### Step 1: Set up the environment

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn torch
```

### Step 2: Run the training script

```bash
cd code
python train_cnn.py
```

### Step 3: Check the results

After running the script, the following files will be generated:
- `testlabel.txt` - Predictions for the test data (in the parent directory)
- `cnn_model.pth` - Saved model weights (in the code directory)

## Model Architecture

The 1D CNN model architecture consists of:

1. **Input Layer**: Takes preprocessed tabular data (14 features after encoding)

2. **Convolutional Blocks** (3 blocks):
   - 1D Convolution with kernel size 3
   - Batch Normalization
   - ReLU Activation
   - Dropout (rate=0.3)

3. **Global Average Pooling**: Reduces the feature dimension

4. **Fully Connected Layers**:
   - Dense layer (64 → 32)
   - ReLU Activation
   - Dropout (rate=0.3)
   - Output layer (32 → 1) with Sigmoid activation

## Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**:
   - '?' values are replaced with NaN
   - Categorical features: filled with mode
   - Numerical features: filled with median

2. **Categorical Encoding**:
   - Label encoding for categorical features
   - Categorical features: workclass, education, marital.status, occupation, relationship, race, sex, native.country

3. **Feature Scaling**:
   - StandardScaler for all features (zero mean, unit variance)

## Training Details

- **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4
- **Loss Function**: Binary Cross-Entropy Loss
- **Batch Size**: 64
- **Early Stopping**: Patience of 15 epochs
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Train/Validation Split**: 80/20 with stratification

## Features

The dataset contains 14 features:

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Working age |
| workclass | Categorical | Type of work |
| fnlwgt | Numerical | Number of observational representatives |
| education | Categorical | Level of education |
| education.num | Numerical | Schooling years |
| marital.status | Categorical | Marital status |
| occupation | Categorical | Occupation type |
| relationship | Categorical | Family relationship |
| race | Categorical | Race |
| sex | Categorical | Gender |
| capital.gain | Numerical | Capital gain |
| capital.loss | Numerical | Capital loss |
| hours.per.week | Numerical | Weekly working hours |
| native.country | Categorical | Country of origin |

## Output

The prediction file `testlabel.txt` contains:
- One prediction per line
- 0: Income ≤ $50K/year
- 1: Income > $50K/year

## Notes

- The model uses GPU if available (CUDA), otherwise falls back to CPU
- Random seed is set to 42 for reproducibility
- Early stopping is used to prevent overfitting

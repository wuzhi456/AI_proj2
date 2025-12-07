# Adult Census Income Prediction

This project implements a neural network-based classifier to predict whether an individual's income exceeds $50K/yr based on census data from the 1994 US Census Bureau database.

## Project Structure

```
AI_proj2/
├── code/
│   ├── model/
│   │   ├── __init__.py
│   │   └── neural_network.py      # Neural network model implementation
│   ├── train/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py  # Data preprocessing utilities
│   │   └── trainer.py             # Training and evaluation utilities
│   └── main.py                     # Main script for training and prediction
├── traindata.csv                   # Training data
├── trainlabel.txt                  # Training labels
├── testdata.csv                    # Test data
├── testlabel.txt                   # Generated test predictions (output)
├── training_notebook.ipynb         # Jupyter notebook for training and visualization
├── report.tex                      # LaTeX report
└── README.md                       # This file
```

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Detailed Package Versions (Recommended)
- torch >= 2.0.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Environment Setup

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n income_pred python=3.8

# Activate environment
conda activate income_pred

# Install dependencies
conda install pytorch numpy pandas scikit-learn matplotlib seaborn jupyter -c pytorch
```

## How to Run

### Method 1: Using the Main Script

To train the model and generate predictions:

```bash
cd code
python main.py
```

This will:
1. Load and preprocess the data
2. Train the neural network model
3. Evaluate the model on validation set
4. Generate predictions on test data
5. Save predictions to `testlabel.txt`

### Method 2: Using Jupyter Notebook (Recommended for Visualization)

To run the training process with detailed visualizations:

```bash
# Start Jupyter Notebook
jupyter notebook

# Open training_notebook.ipynb in the browser
# Run all cells sequentially
```

The notebook will:
1. Load and explore the data
2. Visualize data distribution
3. Preprocess features
4. Train the neural network
5. Display training curves
6. Show evaluation metrics and confusion matrix
7. Generate and save all figures for the report
8. Save predictions to `testlabel.txt`

## Model Architecture

The neural network uses a multi-layer perceptron (MLP) architecture:

- **Input Layer**: Number of features after preprocessing
- **Hidden Layer 1**: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
- **Hidden Layer 2**: 64 neurons + BatchNorm + ReLU + Dropout(0.3)
- **Hidden Layer 3**: 32 neurons + BatchNorm + ReLU + Dropout(0.3)
- **Output Layer**: 1 neuron + Sigmoid activation

### Key Components:
- **Activation Function**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: Batch Normalization and Dropout (rate=0.3)
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary Cross-Entropy Loss
- **Training**: 50 epochs with batch size 128

## Data Preprocessing

1. **Missing Value Handling**:
   - Categorical features: Filled with mode
   - Numerical features: Filled with median

2. **Feature Encoding**:
   - Categorical features: Label encoding

3. **Feature Scaling**:
   - Numerical features: Standardization (z-score normalization)

4. **Train-Validation Split**:
   - 80% training, 20% validation
   - Stratified split to maintain class balance

## Output Files

After running the code:

1. **testlabel.txt**: Contains predicted labels for test data (one label per line)
2. **figures/** (if using notebook):
   - `label_distribution.png`: Distribution of income labels
   - `training_history.png`: Training and validation loss/accuracy curves
   - `confusion_matrix.png`: Confusion matrix visualization
   - `roc_curve.png`: ROC curve and AUC score
   - `test_predictions_distribution.png`: Distribution of test predictions
   - `performance_metrics.csv`: Performance metrics table
   - `model_configuration.csv`: Model configuration parameters

## Generating the Report

To compile the LaTeX report:

```bash
# Install LaTeX distribution (if not already installed)
# On Ubuntu/Debian:
# sudo apt-get install texlive-full

# Compile the report
pdflatex report.tex
pdflatex report.tex  # Run twice for proper references
```

Or use Overleaf:
1. Upload `report.tex` to Overleaf
2. Upload figures from the `figures/` directory
3. Compile using PDFLaTeX

## Troubleshooting

### Common Issues

1. **Import Error for torch**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Missing figures directory**:
   The notebook automatically creates the `figures/` directory. If running the main script, create it manually:
   ```bash
   mkdir figures
   ```

3. **Memory Issues**:
   If you encounter memory errors, reduce the batch size in the code:
   - Edit `code/main.py` or the notebook cell
   - Change `batch_size=128` to `batch_size=64` or `batch_size=32`

4. **CUDA not available**:
   The code automatically falls back to CPU if CUDA is not available. For faster training on GPU:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

All metrics are computed on the validation set during training.

## Project Submission

For project submission, prepare:

1. **report.pdf**: Compiled LaTeX report
2. **code/**: Folder containing all source code
3. **testlabel.txt**: Predictions on test data

Compress into a zip file named: `StudentID_Name.zip` (e.g., `10101010_张三.zip`)

## References

- Dataset: UCI Machine Learning Repository - Adult Census Income
- Framework: PyTorch Deep Learning Framework
- Original Paper: Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers

## Author

Student ID: [Your ID]  
Name: [Your Name]  
Course: CS303 - Artificial Intelligence  
Institution: Southern University of Science and Technology

## License

This project is for academic purposes only.

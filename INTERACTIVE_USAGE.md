# Interactive Training with Visualization

## Overview

The main program (`code/main.py`) now provides **interactive visualization** similar to the Jupyter notebook experience. It displays plots and metrics in real-time during training.

## What You'll See

When you run `python main.py`, the program will:

1. **Data Exploration** - Shows training label distribution with a bar chart
2. **Model Architecture** - Displays the neural network structure
3. **Training Progress** - Prints epoch-by-epoch metrics
4. **Training History** - Plots loss and accuracy curves after training
5. **Confusion Matrix** - Visualizes model performance on validation set
6. **Test Predictions** - Shows distribution of predictions with a chart
7. **Final Summary** - Displays all performance metrics

## How to Run

```bash
cd code
python main.py
```

The program will:
- Load and visualize data
- Train the neural network (50 epochs)
- Display 3 matplotlib windows with visualizations:
  1. Training data distribution
  2. Training/validation curves
  3. Confusion matrix
  4. Test predictions distribution
- Save predictions to `testlabel.txt`

## Features

### Real-time Visualization
- Interactive matplotlib plots appear during execution
- Clear section headers with progress indicators
- Detailed metrics printed to console

### What's Displayed

**During Data Loading:**
```
============================================================
1. Loading and Exploring Data
============================================================
Dataset Statistics:
  Training samples: 22792
  Test samples: 9769
  Number of features: 14

[Bar chart showing label distribution appears]
```

**During Training:**
```
============================================================
4. Training Model
============================================================
Epoch [5/50], Train Loss: 0.3460, Train Acc: 0.8396, ...
Epoch [10/50], Train Loss: 0.3377, Train Acc: 0.8429, ...
...
```

**After Training:**
```
============================================================
5. Training Results Visualization
============================================================
[Line plots showing loss and accuracy curves appear]

============================================================
6. Model Evaluation
============================================================
Validation Metrics:
  Accuracy            : 0.8462
  Precision           : 0.7311
  Recall              : 0.5719
  F1_score            : 0.6418
  Roc_auc             : 0.9035

[Confusion matrix heatmap appears]
```

## Benefits

✅ **Visual Feedback** - See training progress with charts  
✅ **Interactive** - Similar to Jupyter notebook experience  
✅ **Comprehensive** - All metrics and visualizations in one run  
✅ **Easy to Use** - Single command execution  
✅ **Self-contained** - No need to open Jupyter

## Comparison with Jupyter Notebook

| Feature | main.py (Interactive) | training_notebook.ipynb |
|---------|----------------------|-------------------------|
| Visualization | ✅ Automatic | ✅ Step-by-step |
| Execution | Single command | Cell-by-cell |
| Customization | Moderate | High |
| Best for | Quick training | Exploration |

## Note

If running on a server without display, matplotlib will save plots to files instead of showing them interactively. To force non-interactive mode, add this at the top of main.py:

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

Then plots will be saved to the `figures/` directory automatically.

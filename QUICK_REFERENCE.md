# Quick Reference Card

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Run Training
```bash
cd code
python main.py
```
This generates `testlabel.txt` with predictions.

### 3. Generate Figures for Report
```bash
cd code
python generate_figures.py
```
This creates all visualizations in `figures/` directory.

## 📊 What Gets Generated

### Predictions
- **testlabel.txt**: 9,769 predictions (0 or 1)

### Figures (for report)
1. `label_distribution.png` - Training data distribution
2. `training_history.png` - Loss/accuracy curves
3. `confusion_matrix.png` - Performance visualization
4. `roc_curve.png` - ROC curve (AUC=0.9035)
5. `test_predictions_distribution.png` - Test predictions

### Data Files
- `performance_metrics.csv` - Metrics table
- `model_configuration.csv` - Model config

## 📝 Report Compilation

### Using Overleaf (Easiest)
1. Go to [overleaf.com](https://www.overleaf.com)
2. New Project → Upload Project
3. Upload `report.tex`
4. Create `figures/` folder
5. Upload all PNG files from local `figures/` folder
6. Click "Recompile"
7. Download `report.pdf`

### Using Local LaTeX
```bash
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

## 📦 Submission Package

### Create Submission Folder
```bash
# Replace with your actual ID and name
mkdir 10101010_张三
cp report.pdf 10101010_张三/
cp testlabel.txt 10101010_张三/
cp -r code 10101010_张三/

# Create zip
zip -r 10101010_张三.zip 10101010_张三/
```

### What to Include
```
10101010_张三.zip
├── report.pdf          # Compiled LaTeX report
├── testlabel.txt       # Test predictions
└── code/               # All source code
    ├── model/
    ├── train/
    ├── main.py
    └── generate_figures.py
```

## ✅ Pre-Submission Checklist

- [ ] `testlabel.txt` has 9,769 lines
- [ ] `report.pdf` compiled successfully with all figures
- [ ] Code runs without errors
- [ ] Zip file named correctly (StudentID_Name.zip)
- [ ] All three components included (report, code, predictions)

## 🔍 Verification Commands

```bash
# Check predictions
wc -l testlabel.txt  # Should show: 9769

# Verify predictions are binary
sort testlabel.txt | uniq  # Should show: 0 and 1

# Test code
cd code && python main.py  # Should complete successfully
```

## 📈 Expected Performance

- **Accuracy**: ~84-85%
- **ROC-AUC**: ~0.90
- **Training Time**: ~2 minutes on CPU
- **Predictions**: Class 0 (~80%), Class 1 (~20%)

## 🐛 Common Issues

### Import Error
```bash
pip install --upgrade torch numpy pandas scikit-learn matplotlib seaborn
```

### Missing Figures
```bash
cd code
python generate_figures.py
```

### LaTeX Won't Compile
→ Use Overleaf (easiest solution)

## 📁 File Locations

```
AI_proj2/
├── code/               # All source code here
│   ├── model/         # Neural network
│   ├── train/         # Training & preprocessing
│   ├── main.py        # Main script
│   └── generate_figures.py
├── figures/           # Generated visualizations
├── testlabel.txt      # OUTPUT: Predictions
├── report.tex         # LaTeX report source
├── README.md          # Detailed instructions
└── SUBMISSION_GUIDE.md # Submission help
```

## ⏰ Important Dates

- **Deadline**: December 21, 23:55
- **Late Submissions**: NOT ACCEPTED
- **Platform**: Blackboard

## 🎯 Grading

- Report: 70 points
- Code: 20 points
- Predictions: 10 points
- **Total**: 100 points

## 💡 Tips

1. **Run early**: Test everything before deadline
2. **Backup**: Keep multiple copies of submission
3. **Verify**: Check zip file contents before submitting
4. **Time**: Leave time for report compilation
5. **Help**: Read README.md for detailed instructions

## 📧 Resources

- **Detailed Setup**: See `README.md`
- **Submission Help**: See `SUBMISSION_GUIDE.md`
- **Project Overview**: See `PROJECT_SUMMARY.md`

---

**Good luck! 🎓**

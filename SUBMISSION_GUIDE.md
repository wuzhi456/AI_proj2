# Submission Guide

This guide helps you prepare the final submission package for the Adult Census Income project.

## Required Files for Submission

You need to submit three main components:

1. **report.pdf** - Compiled LaTeX report
2. **code/** - All source code
3. **testlabel.txt** - Predictions on test data

## Step-by-Step Submission Preparation

### Step 1: Generate Predictions and Figures

Run the training script to generate predictions:

```bash
cd code
python main.py
```

This will create `testlabel.txt` in the project root.

To generate all figures for the report:

```bash
cd code
python generate_figures.py
```

This will create all figures in the `figures/` directory.

### Step 2: Compile the LaTeX Report

#### Option A: Using Overleaf (Recommended)

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project → Upload Project
3. Upload `report.tex`
4. Create a `figures/` folder in the project
5. Upload all images from the local `figures/` folder:
   - `label_distribution.png`
   - `training_history.png`
   - `confusion_matrix.png`
   - `roc_curve.png`
   - `test_predictions_distribution.png`
6. Compile the project (it will use PDFLaTeX by default)
7. Download the compiled `report.pdf`

#### Option B: Using Local LaTeX Installation

If you have LaTeX installed locally:

```bash
# In the project root directory
pdflatex report.tex
pdflatex report.tex  # Run twice for proper references
```

Note: You may need to run it twice to resolve all references and citations.

### Step 3: Organize Files for Submission

Create the submission structure:

```
[StudentID]_[Name]/
├── report.pdf
├── testlabel.txt
└── code/
    ├── model/
    │   ├── __init__.py
    │   └── neural_network.py
    ├── train/
    │   ├── __init__.py
    │   ├── data_preprocessing.py
    │   └── trainer.py
    ├── main.py
    └── generate_figures.py
```

### Step 4: Create the Submission Archive

#### On Windows:
1. Create a folder named `[StudentID]_[Name]` (e.g., `10101010_张三`)
2. Copy `report.pdf`, `testlabel.txt`, and the entire `code/` folder into it
3. Right-click the folder → Send to → Compressed (zipped) folder
4. Rename the .zip file to match the folder name

#### On macOS/Linux:
```bash
# In the project root directory
mkdir [StudentID]_[Name]
cp report.pdf [StudentID]_[Name]/
cp testlabel.txt [StudentID]_[Name]/
cp -r code [StudentID]_[Name]/
zip -r [StudentID]_[Name].zip [StudentID]_[Name]/
```

Example:
```bash
mkdir 10101010_张三
cp report.pdf 10101010_张三/
cp testlabel.txt 10101010_张三/
cp -r code 10101010_张三/
zip -r 10101010_张三.zip 10101010_张三/
```

## Pre-Submission Checklist

Before submitting, verify:

- [ ] `report.pdf` compiles without errors and contains all sections
- [ ] All figures are visible in the report
- [ ] `testlabel.txt` has exactly 9,769 lines (one prediction per test sample)
- [ ] `code/` directory includes all Python files
- [ ] README.md is included in the code directory for setup instructions
- [ ] The zip file is named correctly: `[StudentID]_[Name].zip`
- [ ] The zip file size is reasonable (should be < 10 MB)

## Verification Commands

To verify your submission:

```bash
# Check testlabel.txt line count
wc -l testlabel.txt
# Should output: 9769 testlabel.txt

# Check predictions are binary (0 or 1)
sort testlabel.txt | uniq
# Should output: 0 and 1

# Verify code runs
cd code
python main.py
# Should complete without errors
```

## Grading Breakdown

- **Report (70 pts)**: Structure, methodology, experiments, writing quality
- **Source Code (20 pts)**: Implementation quality, organization, documentation
- **Predictions (10 pts)**: Accuracy of test predictions

## Important Notes

1. **Deadline**: December 21, 23:55 - **No late submissions accepted**
2. **Plagiarism**: 0% tolerance - discuss but don't copy
3. **File Size**: Keep the submission under 50 MB (should be ~5-10 MB)
4. **Format**: Must be a .zip file, not .rar or .7z
5. **Naming**: Exact format required: `[StudentID]_[Name].zip`

## Troubleshooting

### LaTeX Compilation Errors

If you encounter errors compiling the report:

1. **Missing figures**: Make sure all PNG files are in the `figures/` folder
2. **Package errors**: Use Overleaf which has all packages pre-installed
3. **Special characters**: Ensure proper UTF-8 encoding for Chinese characters

### Test Predictions Issues

If predictions seem incorrect:

1. Verify data files are in the correct location
2. Check that all preprocessing steps completed
3. Ensure model trained for full 50 epochs
4. Verify random seed is set (42) for reproducibility

### Code Won't Run

If code fails to execute:

1. Check Python version (3.8+)
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure you're in the correct directory
4. Check that data files exist: `traindata.csv`, `trainlabel.txt`, `testdata.csv`

## Contact

If you encounter issues that can't be resolved:

1. Check the README.md for detailed setup instructions
2. Review error messages carefully
3. Contact the TA during office hours
4. Post on the course forum (without sharing code)

Good luck with your submission!

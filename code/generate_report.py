"""
Script to generate the project report in PDF format.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os

def create_report():
    # Get the base path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_path, 'report.pdf')
    
    # Create document
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Adult Census Income Prediction Using CNN", title_style))
    story.append(Paragraph("Project 2 Report", styles['Center']))
    story.append(Spacer(1, 30))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    abstract = """
    This project implements a 1D Convolutional Neural Network (CNN) model to predict whether an individual's 
    annual income exceeds $50K based on census data from the 1994 Census bureau database. The model processes 
    14 features including demographic information, education level, occupation, and other relevant attributes. 
    After preprocessing the data and training the CNN model, we achieved a validation accuracy of approximately 
    84.62%. This report details the methodology, implementation, results, and analysis of the project.
    """
    story.append(Paragraph(abstract, body_style))
    story.append(Spacer(1, 20))
    
    # 1. Introduction
    story.append(Paragraph("1. Introduction", heading_style))
    intro = """
    Income prediction is a classic machine learning problem that has significant applications in economics, 
    social science, and policy making. The goal of this project is to develop a classification model that 
    can accurately predict whether an individual's income exceeds $50K per year based on various demographic 
    and socioeconomic features.
    """
    story.append(Paragraph(intro, body_style))
    
    problem = """
    <b>Problem Statement:</b> Given a dataset containing 14 features for each individual including age, 
    education, occupation, marital status, and other attributes, predict whether the individual's annual 
    income exceeds $50K. This is a binary classification problem where the target variable is 0 (income ≤ $50K) 
    or 1 (income > $50K).
    """
    story.append(Paragraph(problem, body_style))
    
    # 2. Dataset Description
    story.append(Paragraph("2. Dataset Description", heading_style))
    dataset_desc = """
    The dataset originates from the 1994 Census bureau database, extracted by Ronny Kohavi and Barry Becker. 
    The training set contains 22,792 samples, and the test set contains 9,769 samples. Each sample has 
    14 features divided into numerical and categorical types.
    """
    story.append(Paragraph(dataset_desc, body_style))
    
    story.append(Paragraph("2.1 Feature Description", subheading_style))
    
    # Feature table
    feature_data = [
        ['Feature', 'Type', 'Description'],
        ['age', 'Numerical', 'Working age of the individual'],
        ['workclass', 'Categorical', 'Type of employment (Private, Government, etc.)'],
        ['fnlwgt', 'Numerical', 'Number of people the census represents'],
        ['education', 'Categorical', 'Highest level of education'],
        ['education.num', 'Numerical', 'Number of years of education'],
        ['marital.status', 'Categorical', 'Marital status of the individual'],
        ['occupation', 'Categorical', 'Type of occupation'],
        ['relationship', 'Categorical', 'Family relationship status'],
        ['race', 'Categorical', 'Race of the individual'],
        ['sex', 'Categorical', 'Gender of the individual'],
        ['capital.gain', 'Numerical', 'Capital gains in a year'],
        ['capital.loss', 'Numerical', 'Capital losses in a year'],
        ['hours.per.week', 'Numerical', 'Weekly working hours'],
        ['native.country', 'Categorical', 'Country of origin']
    ]
    
    table = Table(feature_data, colWidths=[1.3*inch, 1*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # 3. Data Preprocessing
    story.append(Paragraph("3. Data Preprocessing", heading_style))
    preprocess = """
    Data preprocessing is crucial for achieving good model performance. The following steps were applied:
    """
    story.append(Paragraph(preprocess, body_style))
    
    story.append(Paragraph("3.1 Missing Value Handling", subheading_style))
    missing = """
    The dataset contains missing values represented by '?' characters. For categorical features, 
    missing values were imputed using the mode (most frequent value). For numerical features, 
    missing values were imputed using the median value. This approach preserves the distribution 
    characteristics of the data while ensuring no information is lost.
    """
    story.append(Paragraph(missing, body_style))
    
    story.append(Paragraph("3.2 Categorical Feature Encoding", subheading_style))
    encoding = """
    Categorical features were encoded using Label Encoding. Each unique category was assigned 
    a numerical value. The encoder was fitted on the combined training and test data to ensure 
    consistency and handle any categories that might only appear in one set. Eight categorical 
    features were encoded: workclass, education, marital.status, occupation, relationship, 
    race, sex, and native.country.
    """
    story.append(Paragraph(encoding, body_style))
    
    story.append(Paragraph("3.3 Feature Scaling", subheading_style))
    scaling = """
    All features were standardized using StandardScaler to have zero mean and unit variance. 
    This normalization is essential for neural networks as it helps with gradient descent 
    optimization and ensures that features with larger scales don't dominate the learning process.
    """
    story.append(Paragraph(scaling, body_style))
    
    # 4. Model Architecture
    story.append(Paragraph("4. CNN Model Architecture", heading_style))
    model_intro = """
    A 1D Convolutional Neural Network (CNN) was chosen for this classification task. While CNNs 
    are traditionally used for image data, 1D CNNs can effectively capture local patterns in 
    tabular data by treating features as a sequence. The model architecture is designed to 
    extract meaningful representations from the input features through multiple convolutional layers.
    """
    story.append(Paragraph(model_intro, body_style))
    
    story.append(Paragraph("4.1 Architecture Details", subheading_style))
    arch = """
    The CNN model consists of the following components:
    <br/><br/>
    <b>Input Layer:</b> Receives 14 preprocessed features, reshaped to (batch_size, 1, 14) 
    for 1D convolution.
    <br/><br/>
    <b>Convolutional Block 1:</b>
    - 1D Convolution: 64 filters, kernel size 3, padding 1
    - Batch Normalization
    - ReLU Activation
    - Dropout (rate=0.3)
    <br/><br/>
    <b>Convolutional Block 2:</b>
    - 1D Convolution: 128 filters, kernel size 3, padding 1
    - Batch Normalization
    - ReLU Activation
    - Dropout (rate=0.3)
    <br/><br/>
    <b>Convolutional Block 3:</b>
    - 1D Convolution: 64 filters, kernel size 3, padding 1
    - Batch Normalization
    - ReLU Activation
    - Dropout (rate=0.3)
    <br/><br/>
    <b>Global Average Pooling:</b> Reduces spatial dimensions to a single value per channel.
    <br/><br/>
    <b>Fully Connected Layers:</b>
    - Dense layer: 64 → 32 neurons with ReLU activation
    - Dropout (rate=0.3)
    - Output layer: 32 → 1 neuron with Sigmoid activation
    <br/><br/>
    The model contains a total of 52,225 trainable parameters.
    """
    story.append(Paragraph(arch, body_style))
    
    story.append(Paragraph("4.2 Design Rationale", subheading_style))
    rationale = """
    <b>Why CNN for Tabular Data?</b> While traditional machine learning models like Random Forest 
    or XGBoost are commonly used for tabular data, CNNs offer several advantages:
    <br/>
    1. <b>Local Pattern Detection:</b> The convolution operation can detect local feature interactions 
    that might be missed by fully connected networks.
    <br/>
    2. <b>Parameter Sharing:</b> Weight sharing in convolutions provides regularization.
    <br/>
    3. <b>Hierarchical Feature Learning:</b> Multiple layers learn increasingly abstract features.
    <br/><br/>
    <b>Regularization Techniques:</b>
    <br/>
    - <b>Batch Normalization:</b> Stabilizes training and allows higher learning rates
    <br/>
    - <b>Dropout (30%):</b> Prevents overfitting by randomly dropping neurons during training
    <br/>
    - <b>Weight Decay (L2 regularization):</b> Added to the optimizer to prevent overfitting
    """
    story.append(Paragraph(rationale, body_style))
    
    # Page break
    story.append(PageBreak())
    
    # 5. Training Configuration
    story.append(Paragraph("5. Training Configuration", heading_style))
    training = """
    The model was trained with the following configuration:
    """
    story.append(Paragraph(training, body_style))
    
    # Training config table
    config_data = [
        ['Parameter', 'Value'],
        ['Optimizer', 'Adam'],
        ['Learning Rate', '0.001'],
        ['Weight Decay', '1e-4'],
        ['Batch Size', '64'],
        ['Loss Function', 'Binary Cross-Entropy'],
        ['Maximum Epochs', '100'],
        ['Early Stopping Patience', '15 epochs'],
        ['LR Scheduler', 'ReduceLROnPlateau (factor=0.5, patience=5)'],
        ['Train/Val Split', '80% / 20% (stratified)'],
        ['Random Seed', '42']
    ]
    
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 15))
    
    training_detail = """
    <b>Early Stopping:</b> Training was stopped when the validation loss did not improve for 
    15 consecutive epochs. The model with the best validation loss was saved and used for 
    final predictions. In our experiment, training stopped at epoch 73 out of 100.
    <br/><br/>
    <b>Learning Rate Scheduling:</b> The learning rate was reduced by a factor of 0.5 when 
    the validation loss plateaued for 5 epochs, helping the model converge to a better optimum.
    """
    story.append(Paragraph(training_detail, body_style))
    
    # 6. Experimental Results
    story.append(Paragraph("6. Experimental Results", heading_style))
    
    story.append(Paragraph("6.1 Training Progress", subheading_style))
    progress = """
    The model was trained for 73 epochs before early stopping was triggered. The training 
    and validation metrics showed consistent improvement with the following final results:
    <br/><br/>
    - Final Training Loss: 0.3267
    - Final Training Accuracy: 84.77%
    - Final Validation Loss: 0.3219
    - Final Validation Accuracy: 84.62%
    """
    story.append(Paragraph(progress, body_style))
    
    story.append(Paragraph("6.2 Evaluation Metrics", subheading_style))
    
    # Results table
    results_data = [
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ['≤50K', '0.87', '0.93', '0.90', '3461'],
        ['&gt;50K', '0.73', '0.57', '0.64', '1098'],
        ['Weighted Avg', '0.84', '0.85', '0.84', '4559'],
    ]
    
    results_table = Table(results_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("6.3 Confusion Matrix Analysis", subheading_style))
    confusion = """
    The confusion matrix for the validation set:
    <br/><br/>
    - True Negatives (correctly predicted ≤50K): 3,229
    - False Positives (≤50K predicted as >50K): 232
    - False Negatives (>50K predicted as ≤50K): 469
    - True Positives (correctly predicted >50K): 629
    <br/><br/>
    <b>Analysis:</b> The model shows better performance on the majority class (≤50K) with 
    93% recall compared to the minority class (>50K) with 57% recall. This is a common 
    challenge in imbalanced classification problems where the ≤50K class has approximately 
    3x more samples than the >50K class.
    """
    story.append(Paragraph(confusion, body_style))
    
    story.append(Paragraph("6.4 Test Set Predictions", subheading_style))
    test_pred = """
    The trained model generated predictions for all 9,769 test samples:
    <br/><br/>
    - Predicted ≤50K: 7,868 samples (80.5%)
    - Predicted >50K: 1,901 samples (19.5%)
    <br/><br/>
    This distribution is consistent with the expected class distribution in the Adult Census dataset.
    """
    story.append(Paragraph(test_pred, body_style))
    
    # 7. Discussion
    story.append(Paragraph("7. Discussion", heading_style))
    
    story.append(Paragraph("7.1 Model Performance Analysis", subheading_style))
    discussion = """
    The CNN model achieved 84.62% validation accuracy, which is competitive for this dataset. 
    The performance metrics reveal several insights:
    <br/><br/>
    <b>Strengths:</b>
    <br/>
    - High overall accuracy (84.62%)
    - Good precision and recall for the majority class
    - Stable training with gradual improvement
    - Effective regularization preventing significant overfitting
    <br/><br/>
    <b>Challenges:</b>
    <br/>
    - Lower recall for the minority class (>50K) indicating difficulty in identifying 
    high-income individuals
    - Class imbalance affects model's ability to generalize for the minority class
    """
    story.append(Paragraph(discussion, body_style))
    
    story.append(Paragraph("7.2 Comparison with Traditional Methods", subheading_style))
    comparison = """
    While traditional machine learning methods like Random Forest, Gradient Boosting, 
    or Support Vector Machines often achieve similar or slightly higher accuracy on this 
    dataset (typically 85-87%), the CNN approach offers:
    <br/><br/>
    1. Automatic feature learning without extensive feature engineering
    2. Ability to capture non-linear feature interactions
    3. Scalability to larger datasets
    4. Flexibility to incorporate additional data modalities
    """
    story.append(Paragraph(comparison, body_style))
    
    # 8. Limitations and Future Work
    story.append(Paragraph("8. Limitations and Future Work", heading_style))
    
    story.append(Paragraph("8.1 Current Limitations", subheading_style))
    limitations = """
    1. <b>Class Imbalance:</b> The model struggles with the minority class due to the 
    imbalanced dataset (approximately 75% ≤50K vs 25% >50K).
    <br/><br/>
    2. <b>Feature Engineering:</b> The current approach uses basic preprocessing without 
    advanced feature engineering techniques.
    <br/><br/>
    3. <b>Architecture Optimization:</b> The CNN architecture was not extensively tuned 
    through hyperparameter optimization.
    <br/><br/>
    4. <b>Limited Interpretability:</b> Deep learning models are less interpretable compared 
    to traditional methods like decision trees.
    """
    story.append(Paragraph(limitations, body_style))
    
    story.append(Paragraph("8.2 Future Improvements", subheading_style))
    future = """
    1. <b>Address Class Imbalance:</b>
    <br/>
    - Implement class weights in the loss function
    - Use oversampling (SMOTE) or undersampling techniques
    - Apply focal loss to focus on hard examples
    <br/><br/>
    2. <b>Model Enhancements:</b>
    <br/>
    - Experiment with deeper architectures or residual connections
    - Try TabNet or other architectures designed for tabular data
    - Implement attention mechanisms for feature importance
    <br/><br/>
    3. <b>Ensemble Methods:</b>
    <br/>
    - Combine CNN with traditional models (Random Forest, XGBoost)
    - Use model stacking or blending for improved predictions
    <br/><br/>
    4. <b>Advanced Preprocessing:</b>
    <br/>
    - Feature interaction creation
    - Target encoding for categorical variables
    - Polynomial feature generation
    """
    story.append(Paragraph(future, body_style))
    
    # 9. Conclusion
    story.append(Paragraph("9. Conclusion", heading_style))
    conclusion = """
    This project successfully implemented a 1D Convolutional Neural Network for the Adult 
    Census Income prediction task. The model achieved 84.62% validation accuracy, demonstrating 
    the feasibility of using CNN architectures for tabular classification problems.
    <br/><br/>
    The preprocessing pipeline effectively handled missing values, encoded categorical features, 
    and normalized the data for optimal neural network training. The model architecture, combined 
    with regularization techniques (dropout, batch normalization, early stopping), provided 
    stable training without significant overfitting.
    <br/><br/>
    While there is room for improvement, particularly in handling class imbalance and optimizing 
    the architecture, this project demonstrates that CNNs can be effectively applied to structured 
    tabular data classification tasks beyond their traditional image processing applications.
    """
    story.append(Paragraph(conclusion, body_style))
    
    # References
    story.append(Spacer(1, 30))
    story.append(Paragraph("References", heading_style))
    refs = """
    1. Kohavi, R., & Becker, B. (1996). UCI Machine Learning Repository: Adult Data Set. 
    <br/>
    2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
    <br/>
    3. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from 
    overfitting. JMLR, 15(1), 1929-1958.
    <br/>
    4. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training. 
    ICML 2015.
    """
    story.append(Paragraph(refs, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"Report generated: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_report()

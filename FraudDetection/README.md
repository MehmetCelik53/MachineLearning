# IEEE Fraud Detection Project

This project implements a fraud detection system using the IEEE-CIS Fraud Detection dataset. The project includes data optimization, exploratory data analysis, and machine learning modeling using LightGBM.

## Project Structure

```
FraudDetection/
├── notebooks/
│   └── modeling/
│       ├── 01_fixingdtypes.ipynb    # Data type optimization
│       ├── 02_EDA.ipynb              # Exploratory Data Analysis
│       ├── 03_modeling.ipynb         # Model training and evaluation
│       └── 04_FeatureEngineering.ipynb
├── models/                            # Trained model artifacts
├── attributes.txt                     # Dataset description
└── README.md
```

## Dataset

The IEEE-CIS Fraud Detection dataset contains real-world e-commerce transactions collected from a payment service provider and bank fraud detection system.

### Key Features:

- **Transaction Table**: Transaction details including amount, product type, card information, and engineered features
- **Identity Table**: Device and network information (device fingerprinting)
- **800+ Vesta Features**: Engineered features from the fraud detection system

## Workflow

### 1. Data Type Optimization (01_fixingdtypes.ipynb)
- Memory usage analysis
- Float to integer conversion where applicable
- Object to category conversion for low-cardinality features
- Achieved ~50% memory reduction
- Saved optimized data using pickle format to preserve data types

### 2. Exploratory Data Analysis (02_EDA.ipynb)
- Variable categorization (categorical vs numeric)
- Cardinality analysis
- Feature distribution visualization

### 3. Model Training (03_modeling.ipynb)

**Feature Engineering:**
- Label Encoding for low cardinality categoricals
- One-Hot Encoding for numeric features with <10 unique values
- Frequency Encoding for high cardinality features (>50 unique)

**Model:**
- LightGBM Classifier
- Time Series Cross-Validation
- Class weight balancing

**Performance Metrics:**
- ROC-AUC Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Analysis

## Results

The LightGBM model achieves competitive performance on the validation set with proper handling of class imbalance and feature engineering.

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
joblib
```

## Usage

1. Run notebooks in order (01 → 02 → 03)
2. Data files should be placed in `../../data/` directory
3. Model artifacts will be saved in `../../models/` directory

## Notes

- The project uses pickle format for data persistence to maintain optimized data types
- All feature encoders and mappings are saved with the model for reproducibility
- Time-based splitting is used for validation to prevent data leakage

## Author

Mehmet Celik

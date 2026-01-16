#  Customer Churn Prediction - Telco Dataset

##  Project Overview
This project focuses on predicting customer churn for a telecommunications company using advanced machine learning techniques. The goal is to identify customers who are likely to discontinue their services, enabling proactive retention strategies.

##  Dataset
- **Source**: Telco Customer Churn Dataset
- **Size**: 7,043 customers
- **Features**: 20 features including customer demographics, account information, and service usage
- **Target Variable**: Churn (Yes/No)
- **Class Distribution**: 
  - No Churn: 73.46% (5,174 customers)
  - Churn: 26.54% (1,869 customers)

##  Project Workflow

### 1. Data Preprocessing
- **Missing Values Handling**: 
  - 11 missing values in `TotalCharges` column
  - Filled with median value
- **Feature Engineering**:
  - Label encoding for categorical variables
  - Standard scaling for numerical features (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`)
- **Outlier Detection**: 
  - Identified outliers in `SeniorCitizen` and `PhoneService` columns

### 2. Handling Class Imbalance
Multiple techniques were tested:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **BorderlineSMOTE**
- **SMOTETomek** (combination of over-sampling and under-sampling)
- **Class Weights**: Balanced weights in model parameters

### 3. Models Implemented

#### Base Models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**
5. **CatBoost Classifier**
6. **Naive Bayes**

#### Advanced Ensemble:
- **Stacking Classifier** with:
  - Base learners: Random Forest, XGBoost, LightGBM, CatBoost
  - Meta-learner: Logistic Regression

### 4. Hyperparameter Tuning
- **Grid Search CV** for Random Forest
- **Randomized Search CV** for Stacking Ensemble
- **Threshold Optimization** for probability-based predictions (0.20-0.70 range)

##  Results

### Final Model Comparison (Sorted by Accuracy)

| Model | Threshold | Accuracy | F1-Score | Precision | Recall | FN | FP |
|-------|-----------|----------|----------|-----------|--------|----|----|
| **Stacking_Advanced** | 0.50 | **0.7786** | 0.6167 | 0.5705 | 0.6711 | 123 | 189 |
| XGBoost_Optimized | 0.54 | 0.7743 | 0.6169 | 0.5614 | 0.6845 | 118 | 200 |
| CatBoost_Optimized | 0.48 | 0.7658 | 0.6224 | 0.5440 | 0.7273 | 102 | 228 |
| LightGBM_Optimized | 0.52 | 0.7651 | **0.6251** | 0.5422 | **0.7380** | **98** | 233 |
| RandomForest_Optimized | 0.38 | 0.7324 | 0.6077 | 0.4974 | 0.7807 | 82 | 295 |

### Best Model Highlights:
- **Best for Accuracy**: Stacking Advanced (77.86%)
- **Best for F1-Score**: LightGBM Optimized (0.6251)
- **Best for Recall**: LightGBM Optimized (73.80%) - Missed only 98 churners out of 374!

### Soft Voting Ensemble (LGBM + XGB)
- **F1-Score**: 0.6222
- **Precision**: 0.5613
- **Recall**: 0.6979
- **ROC-AUC**: 0.8327
- **Threshold**: 0.54

##  Technologies Used

### Libraries:
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: 
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `catboost`
- **Imbalanced Learning**: `imblearn`

##  Key Insights

1. **Class Imbalance Impact**: SMOTE significantly improved model performance for the minority class
2. **Feature Importance**: Contract type, tenure, and monthly charges were key predictors
3. **Ensemble Power**: Stacking and soft voting ensembles outperformed individual models
4. **Threshold Tuning**: Custom threshold optimization improved F1-scores by 2-5%

##  How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn
```

### Execution
```python
# Load and preprocess data
df = pd.read_csv('telco-Customer-Churn.csv')

# Run the complete pipeline
# (See main notebook/script for detailed implementation)
```


---

**Note**: This project demonstrates the complete machine learning pipeline from data preprocessing to model deployment, with focus on handling imbalanced datasets and optimizing for business metrics.

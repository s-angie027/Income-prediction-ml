# Adult Income Classification with XGBoost, Logistic Regression, and SHAP

# Project Overview
This project builds and evaluates machine learning models to predict whether an individual's income exceeds $50K using the Adult Census Income dataset. The goal was to compare a baseline linear model against a more advanced tree-based model and evaluate performance using classification metrics, ROC-AUC, confusion matrix analysis, and SHAP explainability.

# Business Risk Applications
This project simulates a risk-classification workflow relevant to insurance and financial decision-making. In this context, accurately identifying higher-income individuals can be important for pricing and underwriting. The project also explores the tradeoff between predictive performance and interpretability.

# Models Used
Logistic Regression
XGBoost Classifier

Skills and Methods
- Data cleaning and preprocessing
- One-hot encoding of categorical variables
-Train/test split
- Hyperparameter tuning with GridSearchCV
- Evaluation using:
    -Accuracy
    -Precision
    -Recall
    -F1-score
    -ROC-AUC
    -Confusion matrix
    -Model explainability with SHAP

Key Findings
XGBoost outperformed Logistic Regression across all major metrics.
The largest improvement was on the higher-income class (>50K), where XGBoost achieved better precision, recall, and F1-score.
XGBoost likely performed better because it captures non-linear relationships and feature interactions more effectively than Logistic Regression.
Logistic Regression remained useful as a transparent, interpretable baseline.

## Technologies
- Python
- XGBoost
- SHAP
- Scikit-learn
- Pandas
- matplotlib


## Explainability
SHAP values were used to interpret model predictions and identify the most influential features.

## Feature Importance Plot
![Feature Importance](images/IMG_0033.png)
## Beeswarm Plot
![Beeswarm Plot](images/IMG_0032.png)
## Key Insights
![Key Insights](images/IMG_0031.png)

## Future Improvements
Rebuild preprocessing using Pipeline and ColumnTransformer
Add cross-validation reporting and stronger hyperparameter tuning
Save plots and results automatically
Explore threshold tuning and class imbalance strategies

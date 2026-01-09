# Diabetes Disease Progression Prediction

## Problem Statement

Diabetes is a chronic metabolic disorder affecting millions of people worldwide, characterized by elevated blood glucose levels. Early prediction and monitoring of disease progression are crucial for effective treatment planning and preventing complications. Healthcare providers need accurate tools to predict how diabetes will progress in patients based on their baseline clinical measurements.

**The Challenge:**
Given a patient's baseline clinical measurements including age, sex, body mass index (BMI), average blood pressure, and six blood serum measurements, can we accurately predict the quantitative measure of diabetes disease progression one year after baseline?

This prediction model aims to:
- Enable healthcare providers to identify patients at risk of rapid disease progression
- Support personalized treatment planning and intervention strategies
- Facilitate early preventive measures to slow disease advancement
- Optimize resource allocation in diabetes care management

## Solution Overview

This project implements a **Machine Learning regression model** using **Linear Regression** to predict diabetes disease progression based on ten baseline physiological features. The solution leverages the well-known diabetes dataset from scikit-learn, which contains data from 442 diabetes patients.

### Key Features
- **Predictive Analytics**: Quantitative prediction of disease progression one year after baseline
- **Multi-feature Analysis**: Utilizes 10 standardized clinical features for comprehensive assessment
- **Statistical Modeling**: Linear regression approach for interpretable predictions
- **Performance Metrics**: Comprehensive evaluation using MSE, MAE, RMSE, R², and Adjusted R²

## Dataset Description

### Source
The dataset is sourced from scikit-learn's built-in datasets, originally from:
- **Reference**: Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics
- **URL**: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

### Dataset Characteristics
- **Number of Instances**: 442 patients
- **Number of Features**: 10 baseline variables
- **Target Variable**: Quantitative measure of disease progression one year after baseline (continuous value ranging from 25 to 346)
- **Feature Preprocessing**: All features are mean-centered and scaled by standard deviation × √n_samples

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `sex` | Gender | Binary |
| `bmi` | Body Mass Index | Continuous |
| `bp` | Average Blood Pressure | Continuous |
| `s1` | Total Serum Cholesterol (tc) | Continuous |
| `s2` | Low-Density Lipoproteins (ldl) | Continuous |
| `s3` | High-Density Lipoproteins (hdl) | Continuous |
| `s4` | Total Cholesterol / HDL Ratio (tch) | Continuous |
| `s5` | Log of Serum Triglycerides Level (ltg) | Continuous |
| `s6` | Blood Sugar Level (glu) | Continuous |

### Target Variable
- **Diabetes_Progression**: Quantitative measure of disease progression one year after baseline

### Feature Correlations with Target
Based on the correlation analysis:
- **Strongest positive correlations**: BMI (0.586), s5/triglycerides (0.566), bp (0.441), s4 (0.430)
- **Negative correlation**: s3/HDL (-0.395) - higher HDL is protective
- **Weak correlations**: sex (0.043), s2 (0.174)

## Technical Architecture

### Technology Stack
- **Python**: 3.12.6
- **Core Libraries**:
  - `scikit-learn`: Machine learning model implementation
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `matplotlib`: Data visualization
  - `seaborn`: Statistical data visualization
  - `scipy`: Scientific computing

### Model Pipeline

```
Data Loading → Exploratory Analysis → Feature Engineering → 
Train-Test Split → Standardization → Model Training → 
Evaluation → Prediction
```

## Implementation Details

### 1. Data Loading and Exploration
```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
dataset = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
dataset['Diabetes_Progression'] = diabetes.target
```

### 2. Exploratory Data Analysis
- **Dataset Shape**: 442 samples × 11 features (including target)
- **Missing Values**: None (all features have 442 non-null values)
- **Data Types**: All float64
- **Statistical Summary**: Mean progression ~152, std ~77

### 3. Data Preprocessing
- **Train-Test Split**: 80-20 split (353 training, 89 testing samples)
- **Feature Scaling**: StandardScaler applied to normalize features
- **Random State**: 42 (for reproducibility)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = dataset.drop('Diabetes_Progression', axis=1)
y = dataset['Diabetes_Progression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Model Training
**Algorithm**: Linear Regression
- Simple, interpretable baseline model
- Assumes linear relationship between features and target
- No hyperparameters to tune

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 5. Model Coefficients
The trained model learned the following feature weights:
- Age: 1.75
- Sex: -11.51
- BMI: 25.61 (highest positive impact)
- BP: 16.83
- s1: -44.45 (highest negative impact)
- s2: 24.64
- s3: 7.68
- s4: 13.14
- s5: 35.16 (second highest positive impact)
- s6: 2.35

**Intercept**: 153.74

## Model Performance

### Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Squared Error (MSE)** | 2900.19 | Average squared difference between predicted and actual values |
| **Root Mean Squared Error (RMSE)** | 53.85 | Average prediction error in original units |
| **Mean Absolute Error (MAE)** | 42.79 | Average absolute prediction error |
| **R² Score** | 0.4526 (45.26%) | Model explains ~45% of variance in disease progression |
| **Adjusted R²** | 0.3824 (38.24%) | R² adjusted for number of features |

### Performance Interpretation

**Strengths**:
- The model achieves moderate predictive power (R² = 45.26%)
- RMSE of ~54 units on a scale of 25-346 indicates reasonable accuracy
- Linear model provides interpretable feature importance

**Limitations**:
- R² of 45% suggests significant unexplained variance
- Disease progression is influenced by many factors not captured in the dataset
- Linear assumption may not capture complex non-linear relationships
- Room for improvement with advanced algorithms (Random Forest, Gradient Boosting, Neural Networks)

### Visualization
The project includes an "Actual vs Predicted" scatter plot showing the relationship between true and predicted disease progression values, demonstrating the model's predictive capability and areas of deviation.

## Project Structure

```
Diabetes-prediction/
├── model-training.ipynb    # Main Jupyter notebook with complete analysis
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation (this file)
```

## Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/alimehdi32/Diabetes-prediction.git
cd Diabetes-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook model-training.ipynb
```

## Usage

### Running the Model

1. Open `model-training.ipynb` in Jupyter Notebook
2. Run all cells sequentially (Cell → Run All)
3. The notebook will:
   - Load and explore the diabetes dataset
   - Perform statistical analysis and visualization
   - Train the Linear Regression model
   - Evaluate model performance
   - Display predictions

### Making Predictions

To predict disease progression for a new patient:

```python
# Example: Predict for a patient with standardized features
new_patient = [[0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.020, -0.018]]
prediction = regressor.predict(new_patient)
print(f"Predicted disease progression: {prediction[0]:.2f}")
```

**Note**: Input features must be standardized using the same scaler used during training.

## Results & Insights

### Key Findings

1. **Most Influential Features**:
   - **BMI** (coefficient: 25.61): Strong positive predictor - higher BMI correlates with faster progression
   - **s5/Triglycerides** (coefficient: 35.16): Elevated triglycerides significantly increase progression
   - **BP** (coefficient: 16.83): Higher blood pressure accelerates disease progression

2. **Protective Factors**:
   - **s1/Total Cholesterol** (coefficient: -44.45): Surprisingly negative - may indicate complex lipid interactions
   - **Sex** (coefficient: -11.51): Gender differences in disease progression

3. **Clinical Implications**:
   - Weight management (BMI control) is crucial
   - Lipid profile monitoring (triglycerides, cholesterol) is essential
   - Blood pressure management can slow progression
   - Personalized treatment based on baseline measurements

### Model Limitations & Future Work

**Current Limitations**:
- Linear model may miss non-linear patterns
- Limited to 10 baseline features
- No temporal/longitudinal data
- R² of 45% indicates room for improvement

**Potential Improvements**:
1. **Advanced Algorithms**: 
   - Random Forest Regression
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks
   - Ensemble methods

2. **Feature Engineering**:
   - Polynomial features
   - Interaction terms
   - Domain-specific feature creation

3. **Data Enhancement**:
   - Larger dataset
   - Additional clinical features
   - Temporal progression data
   - Patient lifestyle factors

4. **Model Optimization**:
   - Hyperparameter tuning
   - Cross-validation
   - Feature selection techniques
   - Regularization (Ridge, Lasso)

## Dependencies

```
seaborn
matplotlib
numpy
pandas
scipy
scikit-learn
```

For specific versions, refer to `requirements.txt`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Ali Mehdi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

- **Dataset**: Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004)
- **scikit-learn**: For providing the diabetes dataset and machine learning tools
- **Python Community**: For excellent data science libraries

## Contact

**Author**: Ali Mehdi  
**GitHub**: [@alimehdi32](https://github.com/alimehdi32)  
**Repository**: [Diabetes-prediction](https://github.com/alimehdi32/Diabetes-prediction)

---

## References

1. Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least Angle Regression. *Annals of Statistics*, 32(2), 407-499.
2. Scikit-learn Documentation: [Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
3. Original Data Source: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

---

**Last Updated**: January 2026  
**Version**: 1.0.0

# Software Requirements Specification
## House Price Predictor (ML Project)

---

## 1. Project Overview

**Project Name:** House Price Predictor  
**Repository Name:** `ml-house-price-predictor`  
**Purpose:** Demonstrate end-to-end machine learning workflow and fundamental ML concepts  
**Target Users:** Recruiters assessing ML/data science fundamentals  
**Estimated Development Time:** 4-5 hours

---

## 2. Functional Requirements

### 2.1 Core Features

- **FR1:** Load and explore housing dataset
- **FR2:** Perform exploratory data analysis (EDA) with visualizations
- **FR3:** Preprocess data (handle missing values, outliers, feature scaling)
- **FR4:** Train multiple regression models (Linear, Ridge, Random Forest)
- **FR5:** Evaluate model performance using multiple metrics (R², RMSE, MAE)
- **FR6:** Select best performing model
- **FR7:** Save trained model for reuse
- **FR8:** Make predictions on new housing data
- **FR9:** Visualize feature importance
- **FR10:** Generate comprehensive evaluation report

### 2.2 ML Workflow Stages

```
1. Data Loading
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Preprocessing
   ↓
4. Feature Engineering
   ↓
5. Train-Test Split
   ↓
6. Model Training
   ↓
7. Model Evaluation
   ↓
8. Model Comparison
   ↓
9. Model Persistence
   ↓
10. Prediction on New Data
```

---

## 3. Technical Requirements

### 3.1 Technology Stack

**Core Libraries:**
- **Python:** 3.10+
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computations
- **scikit-learn:** Machine learning algorithms
- **matplotlib:** Basic plotting
- **seaborn:** Statistical visualizations
- **joblib:** Model serialization

**Development Environment:**
- Jupyter Notebook (for exploration)
- Python scripts (for production code)
- Git + GitHub (version control)

### 3.2 Dataset Selection

**Primary Dataset:** California Housing Dataset (built-in to scikit-learn)

**Dataset Characteristics:**
- **Size:** 20,640 samples
- **Features:** 8 numerical features
- **Target:** Median house value (in $100,000s)
- **No Missing Values:** Clean dataset, good for learning
- **Real-World Data:** Based on 1990 California census

**Features:**
1. `MedInc` - Median income in block group
2. `HouseAge` - Median house age in block group
3. `AveRooms` - Average number of rooms per household
4. `AveBedrms` - Average number of bedrooms per household
5. `Population` - Block group population
6. `AveOccup` - Average number of household members
7. `Latitude` - Block group latitude
8. `Longitude` - Block group longitude

**Target Variable:**
- `MedHouseVal` - Median house value (in $100,000s)

**Alternative Dataset:** Kaggle House Prices (more features, requires download)

---

### 3.3 Data Preprocessing Pipeline

#### 3.3.1 Exploratory Data Analysis

```python
# EDA Steps
1. Load dataset
2. Display basic statistics (mean, std, min, max)
3. Check data types
4. Identify missing values (none expected)
5. Detect outliers using IQR method
6. Visualize distributions (histograms)
7. Correlation matrix heatmap
8. Scatter plots for key relationships
```

#### 3.3.2 Data Cleaning

```python
# Outlier Handling Strategy
- Identify outliers using IQR (Interquartile Range)
- Cap outliers at 99th percentile (not remove)
- Rationale: Preserve data while reducing extreme values
```

#### 3.3.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Why Scaling?**
- Linear/Ridge regression benefit from scaled features
- Prevents features with large ranges from dominating
- Random Forest doesn't require scaling (but doesn't hurt)

#### 3.3.4 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 80/20 split
    random_state=42    # Reproducibility
)
```

---

### 3.4 Model Selection & Training

#### 3.4.1 Models to Compare

| Model | Reason for Inclusion | Expected Performance |
|-------|---------------------|---------------------|
| **Linear Regression** | Baseline, simple, interpretable | R² ~ 0.60 |
| **Ridge Regression** | Handles multicollinearity | R² ~ 0.61 |
| **Random Forest** | Captures non-linear patterns | R² ~ 0.80 |

#### 3.4.2 Training Code Structure

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Dictionary of models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = evaluate_model(y_test, y_pred)
```

---

### 3.5 Evaluation Metrics

#### 3.5.1 Metrics Definitions

```python
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

def evaluate_model(y_true, y_pred):
    return {
        'R² Score': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
```

#### 3.5.2 Metric Interpretation

| Metric | Description | Good Value |
|--------|-------------|------------|
| **R² Score** | Variance explained (0-1) | > 0.70 |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **MAPE** | Mean Absolute % Error | < 20% |

**Success Threshold:** R² > 0.60 for MVP (Linear Regression baseline)

---

### 3.6 Feature Importance Analysis

```python
# For Random Forest (best model)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize top 10 features
plt.barh(feature_importance['feature'][:10], 
         feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features')
```

**Expected Important Features:**
1. MedInc (median income) - strongest predictor
2. Latitude / Longitude (location matters)
3. AveRooms (house size indicator)

---

### 3.7 Model Persistence

```python
import joblib

# Save best model and scaler
joblib.dump(best_model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Load for predictions
loaded_model = joblib.load('models/house_price_model.pkl')
loaded_scaler = joblib.load('models/scaler.pkl')
```

---

## 4. Project Structure

```
ml-house-price-predictor/
├── data/
│   ├── raw/                    # Original dataset (if not using sklearn)
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA and visualization
│   ├── 02_preprocessing.ipynb  # Data cleaning
│   └── 03_modeling.ipynb       # Model training
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load dataset
│   ├── preprocessor.py         # Preprocessing functions
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation functions
│   └── predict.py              # Prediction script
├── models/
│   ├── house_price_model.pkl   # Saved model
│   └── scaler.pkl              # Saved scaler
├── outputs/
│   ├── figures/                # Plots and visualizations
│   └── reports/                # Evaluation reports
├── tests/
│   └── test_preprocessing.py   # Unit tests (optional)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Key Code Components

### 5.1 Data Loading (`data_loader.py`)

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_housing_data():
    """Load California housing dataset"""
    housing = fetch_california_housing(as_frame=True)
    
    df = housing.frame
    X = housing.data
    y = housing.target
    
    return df, X, y, housing.feature_names
```

### 5.2 Preprocessing (`preprocessor.py`)

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

class HousingPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def handle_outliers(self, df, columns, percentile=99):
        """Cap outliers at specified percentile"""
        df_copy = df.copy()
        for col in columns:
            cap = np.percentile(df_copy[col], percentile)
            df_copy[col] = df_copy[col].clip(upper=cap)
        return df_copy
    
    def fit_transform(self, X_train):
        """Fit scaler and transform training data"""
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X_test):
        """Transform test data using fitted scaler"""
        return self.scaler.transform(X_test)
```

### 5.3 Training Script (`train.py`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    # Load data
    df, X, y, feature_names = load_housing_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess
    preprocessor = HousingPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    score = model.score(X_test_scaled, y_test)
    print(f"Test R² Score: {score:.4f}")
    
    # Save model
    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    
    return model, score

if __name__ == '__main__':
    train_model()
```

### 5.4 Prediction Script (`predict.py`)

```python
import joblib
import numpy as np

def predict_price(features):
    """
    Predict house price for new data
    
    Args:
        features: dict with keys matching feature names
                  e.g., {'MedInc': 3.5, 'HouseAge': 20, ...}
    
    Returns:
        Predicted price in $100,000s
    """
    # Load model and scaler
    model = joblib.load('models/house_price_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Convert to array in correct order
    feature_array = np.array([[
        features['MedInc'],
        features['HouseAge'],
        features['AveRooms'],
        features['AveBedrms'],
        features['Population'],
        features['AveOccup'],
        features['Latitude'],
        features['Longitude']
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(feature_array)
    prediction = model.predict(features_scaled)[0]
    
    return prediction

# Example usage
if __name__ == '__main__':
    sample_house = {
        'MedInc': 3.5,
        'HouseAge': 25,
        'AveRooms': 6.0,
        'AveBedrms': 1.0,
        'Population': 1200,
        'AveOccup': 3.0,
        'Latitude': 34.05,
        'Longitude': -118.25
    }
    
    price = predict_price(sample_house)
    print(f"Predicted price: ${price * 100000:,.2f}")
```

---

## 6. Visualization Requirements

### 6.1 EDA Visualizations

**Must Include:**
1. **Distribution Plots:** Histogram for each feature
2. **Correlation Heatmap:** Show feature relationships
3. **Scatter Plots:** 
   - Price vs MedInc (strongest correlation)
   - Price vs Location (lat/long)
4. **Box Plots:** Detect outliers visually
5. **Geographic Plot:** Price distribution on California map

**Example Code:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('outputs/figures/correlation_heatmap.png')

# Price vs Income scatter
plt.figure(figsize=(8, 6))
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('House Price vs Income')
plt.savefig('outputs/figures/price_vs_income.png')
```

### 6.2 Model Performance Visualizations

**Must Include:**
1. **Actual vs Predicted Plot:** Scatter plot showing prediction accuracy
2. **Residual Plot:** Errors distribution
3. **Feature Importance:** Bar chart for Random Forest
4. **Model Comparison:** Bar chart comparing R² scores

---

## 7. Non-Functional Requirements

### 7.1 Code Quality

- **Modularity:** Separate data loading, preprocessing, training, evaluation
- **Documentation:** Docstrings for all functions
- **Type Hints:** Use where helpful (optional for MVP)
- **PEP 8 Compliance:** Consistent code style
- **Reproducibility:** Fixed random seeds (42)

### 7.2 Performance

- **Training Time:** < 5 minutes on average laptop
- **Prediction Time:** < 100ms for single prediction
- **Memory Usage:** < 2GB RAM during training

### 7.3 Reproducibility

```python
# Set all random seeds
import random
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use in all random operations
train_test_split(X, y, random_state=RANDOM_SEED)
RandomForestRegressor(random_state=RANDOM_SEED)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (Optional)

```python
# tests/test_preprocessing.py
import unittest
import numpy as np
from src.preprocessor import HousingPreprocessor

class TestPreprocessing(unittest.TestCase):
    def test_scaler_transforms_correctly(self):
        preprocessor = HousingPreprocessor()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = preprocessor.fit_transform(X_train)
        
        # Check mean is close to 0
        self.assertAlmostEqual(X_scaled.mean(), 0, places=10)
        
        # Check std is close to 1
        self.assertAlmostEqual(X_scaled.std(), 1, places=1)
```

### 8.2 Model Validation Checklist

- [ ] R² score > 0.60 on test set
- [ ] RMSE is reasonable (< 1.0 for scaled data)
- [ ] Model generalizes (train R² ≈ test R²)
- [ ] Predictions are in valid range (0-5 for $0-500k)
- [ ] Feature importance makes intuitive sense
- [ ] Model can be saved and loaded successfully
- [ ] Predictions reproducible with same seed

---

## 9. Out of Scope (Future Enhancements)

- ❌ Deep learning models (neural networks)
- ❌ Advanced feature engineering (polynomial features, interactions)
- ❌ Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- ❌ Cross-validation (k-fold CV)
- ❌ Ensemble methods beyond Random Forest
- ❌ Real-time predictions API
- ❌ Web interface for predictions
- ❌ Time series analysis (price trends)
- ❌ Geospatial analysis (mapping libraries)

These can be added as "Future Improvements" in README.

---

## 10. Deliverables

### 10.1 Required Files

1. **Jupyter Notebooks:**
   - Exploration and EDA
   - Model training and evaluation
   - Well-documented with markdown cells

2. **Python Scripts:**
   - Production-ready training script
   - Prediction script for new data

3. **Saved Models:**
   - `house_price_model.pkl`
   - `scaler.pkl`

4. **Visualizations:**
   - Correlation heatmap
   - Feature importance plot
   - Actual vs Predicted plot
   - Model comparison chart

5. **Documentation:**
   - Comprehensive README
   - requirements.txt
   - Code comments

### 10.2 Evaluation Report

**Generate Markdown Report:**
```markdown
# House Price Prediction - Model Evaluation Report

## Dataset Summary
- Total Samples: 20,640
- Features: 8
- Target: Median House Value

## Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.606 | 0.729 | 0.533 |
| Ridge Regression | 0.606 | 0.729 | 0.533 |
| **Random Forest** | **0.812** | **0.503** | **0.328** |

## Best Model: Random Forest
- Selected based on highest R² score
- Captures non-linear relationships
- Robust to outliers

## Feature Importance (Top 5)
1. MedInc (Median Income): 0.52
2. Longitude: 0.18
3. Latitude: 0.15
4. AveRooms: 0.08
5. HouseAge: 0.04

## Conclusion
Random Forest achieves R² = 0.812, explaining 81% of variance in house prices.
Median income is the strongest predictor, followed by location.
```

---

## 11. Success Criteria

The project is considered successful when:

✅ Model achieves R² > 0.60 on test set  
✅ Code demonstrates complete ML pipeline  
✅ Evaluation metrics clearly presented  
✅ Model can make predictions on new data  
✅ README explains methodology clearly  
✅ Visualizations effectively communicate insights  
✅ Code is modular and well-documented  
✅ Model and scaler saved for reuse  
✅ Results are reproducible (fixed random seed)

---

## 12. README Requirements

The project README must include:

1. **Project Overview:** Problem statement and approach
2. **Dataset Description:** Features and target variable
3. **Tech Stack:** Python libraries used
4. **Methodology:**
   - Data preprocessing steps
   - Models compared
   - Evaluation metrics
5. **Results:** Performance comparison table
6. **Key Findings:** Feature importance insights
7. **How to Run:**
   - Install dependencies
   - Run training script
   - Make predictions
8. **Visualizations:** Include key plots
9. **Challenges & Learnings:** What you learned
10. **Future Improvements:** Potential enhancements

---

## 13. Portfolio Value

**Signal This Sends:**
> "I understand the complete machine learning workflow from data exploration to model deployment, and can communicate my approach clearly."

**Key Talking Points for Interviews:**
- Why you chose Random Forest over Linear Regression
- How you handled outliers and why
- Your approach to feature scaling
- Train-test split rationale (why 80/20)
- How you would improve the model (hyperparameter tuning, more features)
- Real-world deployment considerations
- Handling concept drift (data changes over time)
- Interpretability vs accuracy trade-offs

---

## 14. Quick Start Commands

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run training
python src/train.py

# 3. Make prediction
python src/predict.py

# 4. Explore in Jupyter
jupyter notebook notebooks/01_exploration.ipynb
```

**Requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
jupyter>=1.0.0
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Ready for Implementation

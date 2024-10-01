# 🗳️ Machine Learning Analysis of Voting Tendencies by Electoral District, Income-Based Voter Behavior in Washington State
This project aims to predict voter behavior using income data from Washington State through machine learning models, exploring how socio-economic factors influence voting patterns.

## 📁 Project Overview
This study employs various machine learning models to analyze voting patterns in Washington State, focusing on the impact of income data. The project explores models such as linear regression, polynomial regression with ridge regularization, random forests, and support vector regression. Different evaluation metrics such as RMSE, MAE, and R² are used to determine the most effective model for predicting voter behavior.

## 🗳️ Key Objectives
-  **Build models** to predict voter behavior based on income levels.
-  **Collaborate with team members** to analyze additional factors such as age, gender, and education.
-  **Implement a batch-average loss function** to train models without individual-level voting data.

## 📑 Methodology
### 1. Data Collection
-  **Income Data**: Collected from the U.S. Census Bureau.
-  **Voting Data**: Sourced from Harvard’s voting records database.
-  **Geospatial Data**: Acquired using GIS and GeoPandas.

### 2. Preprocessing
-  **Standardization**: Cleaned and standardized datasets using Pandas.
-  **Merging**: Combined income data with voting data across Washington State.

### 3. Model Development
-  **Models**: Includes `SimpleNN`, `DeepNN`, and traditional algorithms like `Random Forest` and `SVR`.
-  **Optimizers**: Models were fine-tuned using `SGD` and `Adam` optimizers.

### 4. Evaluation
-  **Metrics**: Evaluated using RMSE, MAE, and R².
-  **Cross-validation**: Applied for model robustness.

# 📊 Key Findings
-  **Random Forest**: Emerged as the top model with the highest predictive accuracy (RMSE: 0.0796, R²: 0.88).
-  **Batch-Average Loss Function**: Enabled accurate predictions without requiring individual-level data.
-  **Correlations**: Identified significant correlations between income and voting patterns in Washington State, applicable to other regions like Texas and New York.

# 🛠 Tools Used
-  **Python**: Programming language.
-  **Pandas**: Data cleaning and preprocessing.
-  **GeoPandas**: Geospatial analysis.
-  **Scikit-learn**: Machine learning models.
-  **PyTorch**: Neural network development.

# 📝 Conclusion
This project demonstrated the power of machine learning in analyzing socio-economic factors like income on voter behavior. Future research could incorporate additional demographic factors and refine model accuracy through enhanced data integration.


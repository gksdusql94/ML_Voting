# 🗳️ Machine Learning Analysis of Voting Tendencies by Electoral District  
**Income-Based Voter Behavior in Washington State**

**Author:** Yeonbi Han  

## 📁 Abstract  
This project investigates voter behavior using income data from Washington State, employing machine learning to predict voting patterns. The model integrates geographical and income data, improving voter prediction models using batch processing. Collaborating with team members focusing on gender, education, and age-based studies in other states, this project aims to inform policymakers by providing insights into complex electoral dynamics.

---

## 🗂️ Files and Methodology Overview

### 1. `240202 NY_YB_MatchTractsWithPrecincts_CentroidMethod.ipynb` & `240209 WA_MatchTractsWithPrecincts_CentroidMethod.ipynb`
- Utilizes GIS to align electoral precinct centroids with census tracts from NY and WA states, helping in analyzing voter behaviors at a geographic level.

### 2. `240223Filtered Data.ipynb` & `240223Final Data.ipynb`
- Extracts and filters income data categorized by income levels from Washington State’s census tracts, prepping it for machine learning analysis.

### 3. `240308 YB Voting_modeling.ipynb`
- Implements a basic neural network model (SimpleNN) with income data to predict voting outcomes using preprocessing and normalization techniques.

### 4. `240322_different_ml.ipynb`
- Compares models like Linear Regression, Polynomial Regression, Random Forest, and SVR. The Random Forest model delivered the best performance with an RMSE of 0.0797.

### 5. `240322_improved_deep_voting_model.ipynb`
- Improved deep learning model (DeepNN) with batch normalization and dropout layers to prevent overfitting and capture complex relationships between income and voting patterns.

### 6. `240417_different_eval.ipynb`
- Introduces varied evaluation metrics (RMSE, MAE, R²) and cross-validation methods to ensure the model's robustness and improve overall result accuracy.

---

## 🧪 Methodology Overview

###  Data Collection and Integration
- Geospatial (GIS) and income data from the U.S. Census and WA voting records are integrated using centroid mapping to match census tracts with precincts, allowing for enhanced analysis of voting patterns.

###  Model Development and Testing
1. **Linear Regression & Polynomial Regression**: Simple models as a baseline.
2. **Random Forest**: Best performance, capturing nonlinear relationships in voting patterns.
3. **SVR (Support Vector Regression)**: Moderate performance, capturing nonlinearity.
4. **Improved Deep Learning (DeepNN)**: Incorporates advanced layers to prevent overfitting and capture intricate patterns in data.

###  Evaluation Metrics
- Metrics such as RMSE, MAE, and R² are used to compare model performances, while cross-validation ensures generalization across different data subsets.

---

##  Results
- The **Random Forest model** provided the best accuracy with the lowest RMSE of **0.0797**, followed by SVR and Polynomial Regression. The study demonstrates that integrating income and geospatial data through machine learning leads to effective voting behavior predictions.

---

##  Challenges
- **GIS Integration**: Aligning different datasets from NY and WA required significant GIS manipulation.
- **Overfitting**: Initial deep learning models with many layers suffered from overfitting, prompting model simplification.
- **Income Binning**: Grouping income data into larger intervals improved interpretability without compromising on the granularity.

---

## 🏁 Conclusion
The **Random Forest model** was most effective in predicting voter behaviors based on income. This framework can be extended by incorporating more demographic variables (e.g., education, age) and applying it to other states to yield broader insights into electoral patterns.

---

## 📚 References
- **U.S. Census Bureau**  
- **Harvard University Database**  
- **GIS Resources**


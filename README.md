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
Geospatial and income data were collected from the U.S. Census and voting records from Washington State. GIS techniques (centroid mapping) were used to match electoral precincts with census tracts. After extensive data cleaning and standardization, the final dataset incorporated income levels and election results across various precincts.

```python
import geopandas as gpd # 1. Importing Data and Libraries
import pandas as pd

# Load census tract and precinct data
census_blocks = gpd.read_file('zip://path_to_census_blocks')
precincts = gpd.read_file('zip://path_to_precincts')

# Align geospatial reference systems
precincts = precincts.to_crs(census_blocks.crs)

# Find intersecting precincts for a given census block # 2. Calculate Overlap Percentages for Census Tracts and Precincts
census_block_polygon = census_blocks.loc[0, 'geometry']
overlap_precincts = precincts[precincts['geometry'].intersects(census_block_polygon)]

# Calculate overlap percentage
for idx, row in overlap_precincts.iterrows():
    intersection_area = row['geometry'].intersection(census_block_polygon).area
    overlap_precincts.at[idx, 'overlap_percentage'] = (intersection_area / row['geometry'].area) * 100
```
###  Model Development and Testing
1. **Linear Regression & Polynomial Regression**: Simple models as a baseline.
2. **Random Forest**: Best performance, capturing nonlinear relationships in voting patterns.
3. **SVR (Support Vector Regression)**: Moderate performance, capturing nonlinearity.

4. **Simple Deep Learning (DeepNN)**: Incorporates advanced layers to prevent overfitting and capture intricate patterns in data.

```python
import torch.nn as nn
import torch.optim as optim
# Define SimpleNN class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
simple_model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(simple_model.parameters(), lr=0.001)
```

5. **Improved Deep Learning (DeepNN)**

```python
# Define DeepNN class with BatchNorm and Dropout
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
```
# Initialize improved deep learning model
deep_model = DeepNN()

###  Evaluation Metrics
- Metrics such as RMSE, MAE, and R² are used to compare model performances, while cross-validation ensures generalization across different data subsets.
```python
# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item()}')

# Evaluation
y_pred = model(X_test).detach().numpy()
rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
print(f'RMSE: {rmse}')
```
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


# 🗳️ Machine Learning Analysis of Voting Tendencies by Electoral District  
**Income-Based Voter Behavior in Washington State**

**Author:** Yeonbi Han  

## 📁 Abstract  
This project investigates voter behavior using income data from Washington State, employing machine learning to predict voting patterns. The model integrates geographical and income data, improving voter prediction models using batch processing. Collaborating with team members focusing on gender, education, and age-based studies in other states, this project aims to inform policymakers by providing insights into complex electoral dynamics.

---

## 🗂️ Files and Methodology Overview

### 1. `240202 NY_YB_MatchTractsWithPrecincts_CentroidMethod.ipynb` & `240209 WA_MatchTractsWithPrecincts_CentroidMethod.ipynb`
- Utilizes GIS to align electoral precinct centroids with census tracts from NY and WA states, helping in analyzing voter behaviors at a geographic level.
```python
base_map = census_blocks.plot(column='relative_error',legend=True, figsize=(12, 8))
```
![image](https://github.com/user-attachments/assets/57423fb1-9309-4ffc-8ae1-1771e8005d40)

### 2. `240223Filtered Data.ipynb` & `240223Final Data.ipynb`
- Extracts and filters income data categorized by income levels from Washington State’s census tracts, prepping it for machine learning analysis.

### 3. `240308 YB Voting_modeling.ipynb`
- Implements a basic neural network model (SimpleNN) with income data to predict voting outcomes using preprocessing and normalization techniques.

### 4. `240322_different_ml.ipynb`
- Compares models like Linear Regression, Polynomial Regression, Random Forest, and SVR. The Random Forest model delivered the best performance with an RMSE of 0.0797.

### 5. `240322_improved_deep_voting_model.ipynb`
This enhanced deep neural network incorporates batch normalization and dropout layers to prevent overfitting while capturing complex relationships between income and voting patterns. Various optimizers, such as ADAM and ADAGRAD, were tested to enhance model performance.

```python
optimizer2 = optim.Adam(model.parameters(), lr=0.001) #New: SGD ->> ADAM
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001) : Original
#optimizer = optim.Nadam(model.parameters(), lr=0.001)
#optimizer = optim.RMSprop(model.parameters(), lr=0.001)

import torch
num_epochs = 500
losses = []
for epoch in range(num_epochs):
    for i,(train_X, train_y)  in enumerate(county_dataloader):
      epoch_loss = 0.0
      if i<32:

        # Convert lists to PyTorch tensors
        features = torch.tensor(train_X)
        target = torch.tensor(train_y)

        # Forward pass, loss computation, and backward pass
        outputs = model(features)
        # print(outputs) # issue here
        loss = custom_loss(outputs, target)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        epoch_loss += loss.item()
        average_epoch_loss = epoch_loss /50
       # losses.append(loss.item())
        losses.append(average_epoch_loss)
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
```

### 6. `240417_different_eval.ipynb`
- Introduces varied evaluation metrics (RMSE, MAE, R²) and cross-validation methods to ensure the model's robustness and improve overall result accuracy.

---

## 🧪 Methodology Overview

###  Data Collection and Integration
Geospatial and income data were collected from the U.S. Census and voting records from Washington State. GIS techniques (centroid mapping) were used to match electoral precincts with census tracts. After extensive data cleaning and standardization, the final dataset incorporated income levels and election results across various precincts.

```python
import geopandas as gpd # 1. Importing Data and Libraries
import pandas as pd

# Filter precincts with overlap_percentage >= 50
filtered_precincts = overlap_precincts[overlap_precincts['overlap_percentage'] >= 50]

# Now, filtered_precincts contains only the precincts with overlap_percentage >= 50
# Visualize the selected census block and precincts that are approximating it
census_blocks.loc[[CENSUS_BLOCK_INDEX]].explore('STATEFP')
```
![image](https://github.com/user-attachments/assets/7c8a8cf1-418f-479b-b5fa-c0a9dc3c8e6d)
```python
# Visualize the filtered precincts with their Biden proportion
filtered_precincts.explore('Biden_proportion')
```
![image](https://github.com/user-attachments/assets/ea56c1e7-5f92-43a2-80e2-976eddfb9a3e)

```python
# Show the approximation error
base = census_blocks.loc[[CENSUS_BLOCK_INDEX]].plot('STATEFP', color='red', alpha=0.5)
filtered_precincts.plot('STATEFP', color='yellow', ax=base, alpha=0.5)
```
![image](https://github.com/user-attachments/assets/c154008d-4aef-4d46-ab8a-cb164e17acc5)


###  Model Development and Testing
1. **Linear Regression & Polynomial Regression**: Simple models as a baseline.
```python
from sklearn.linear_model import LinearRegression # LInear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_np, y_train_np)
```

2. **Random Forest**: Best performance, capturing nonlinear relationships in voting patterns.
```python
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_np, y_train_np)
```

3. **SVR (Support Vector Regression)**: Moderate performance, capturing nonlinearity.

```python
from sklearn.svm import SVR # Support Vector Regressor(SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_np, y_train_np)
```

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
plt.plot(losses[10:], label='Training Loss') #The code plots only the data points for the training loss after excluding the first 10 epochs.
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation
y_pred = model(X_test).detach().numpy()
rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
print(f'RMSE: {rmse}')
```
![image](https://github.com/user-attachments/assets/ab9c81c6-e07d-4d3d-8713-86c39c0fd773)

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


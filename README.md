# US President Precinct Voting Modeling (ML by State)

## Project Overview
This project aims to develop and evaluate machine learning models for predicting voting patterns across different electoral districts, focusing on **income-based voter behavior** in **Washington State**. The project combines income data, geographical information, and voting records to build a predictive model capable of analyzing trends across various demographics and states.

## Key Objectives
- Build a machine learning model to predict voter behavior based on **income levels**.
- Collaborate with team members to integrate additional variables such as **age**, **gender**, and **education**, analyzing how these factors impact voting patterns in different regions, including **Texas** and **New York**.
- Compare the performance of various machine learning algorithms, including **Random Forest**, **Linear Regression**, **Support Vector Regression (SVR)**, and others, to identify the most accurate model.
- Design and implement a **batch-average loss function** to predict voting outcomes without relying on individual-level voting data.

## Methodology
### 1. Data Collection
   - **Income Data**: Collected through U.S. Census Bureau and filtered to include income levels across Washington's census tracts.
   - **Voting Data**: Voting records obtained from Harvard's database and electoral precinct data from the **U.S. Census Bureau**.
   - **Geographical Data**: Incorporated geospatial data using **GIS** and **GeoPandas** to map electoral districts to voting precincts.

### 2. Data Preprocessing
   - Cleaned and filtered datasets using **Python** and **Pandas**.
   - Handled missing values and standardized data for effective model training.
   - Merged income and voting datasets for comprehensive analysis.

### 3. Model Development
   - Developed two machine learning models: **SimpleNN** (basic neural network) and **DeepNN** (multi-layer neural network) to predict voting percentages based on income levels.
   - Implemented **SGD** and **Adam** optimizers to fine-tune model performance.
   - Applied **Random Forest**, **Linear Regression**, and **SVR** models to explore other approaches for predicting voting patterns.

### 4. Evaluation Metrics
   - Evaluated model performance using **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)** metrics.
   - Performed cross-validation to ensure generalizability of the models.

## Key Findings
- The **Random Forest** model showed the highest predictive accuracy with the lowest RMSE (0.0796) and the highest R² value (0.88), making it the best fit for this electoral dataset.
- The **batch-average loss function** provided an innovative solution for training the model without individual voting data.
- The analysis revealed significant correlations between **income levels** and voting patterns in Washington State, with potential applications in other states like Texas and New York.

## Tools Used
- **Python**: Data processing, model development, and evaluation.
- **Pandas**: Data cleaning and manipulation.
- **GeoPandas/GIS**: Geographic analysis and mapping electoral districts.
- **Scikit-learn**: Model implementation (Random Forest, Linear Regression, SVR).
- **PyTorch**: Neural network model development (SimpleNN, DeepNN).

## Conclusion
This project successfully implemented machine learning models to predict voter behavior based on socio-economic factors. The use of **Random Forest** and the **batch-average loss function** proved effective in understanding voting tendencies without individual-level data. Future improvements include incorporating more demographic factors and refining model accuracy through enhanced data integration.

Integrated income data with geographic information to predict voter behavior and improve model accuracy using batch processing to combine various data sources.
-	Collaborated with team members to analyze variables such as gender, age, and education in other states, including Texas, to broaden the understanding of voting patterns across regions.
-	Utilized Python and GIS tools to calculate geographic centroids for electoral districts, aligning them with precincts, and analyzed the correlation between income levels and voting outcomes.
-	Compared various ML algorithms, including Random Forest, Linear Regression, and SVR. The Random Forest model showed the highest predictive accuracy with an RMSE of 0.0797, outperforming other models.




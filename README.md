# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

COMPANY : CODTECH IT SOLUTION

NAME : PUNEETH H LAMANI

DOMAIN : DATA ANALYST INTERN

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH


# Description

The provided Python script builds a predictive model to estimate the 2023 GDP of different countries using historical GDP data from 1960 to 2022. It starts by importing necessary libraries (pandas, numpy, matplotlib, seaborn, and sklearn) and loading a CSV dataset containing GDP values. The script preprocesses the data by dropping irrelevant columns, handling missing values using forward and backward filling, and ensuring all numerical columns are properly formatted. It then separates the dataset into features (GDP values from 1960-2022) and the target variable (2023 GDP). Missing values in the features are imputed using the column mean. The data is then split into training (80%) and testing (20%) sets, and a Linear Regression model is trained to learn patterns from historical GDP values. After training, the model predicts GDP values for the test set, and its performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score. The script also allows users to input a country name, retrieves the relevant data, imputes missing values, and predicts the country's GDP for 2023. The actual GDP trend from 1960 to 2022 is plotted along with the predicted 2023 GDP, providing a visual representation of the model’s estimation.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset

df = pd.read_csv(rAlbania
"C:\Users\joyfu.DESKTOP-5ETBPHM\OneDrive\Desktop\puneeth\codtech internship\task 3\All Countries and Economies GDP (US) 1960-2023.csv")

# Keep country names for reference
df["Country Name"] = df["Country Name"].astype(str)  # Ensure it's a string

# Drop unnecessary columns
columns_to_drop = ["Country Code", "Indicator Name", "Indicator Code"]
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Drop rows where GDP for 2023 is missing (our target variable)
df_cleaned = df_cleaned.dropna(subset=["2023"], how='all')

# Fill missing values using forward fill and backward fill
df_cleaned = df_cleaned.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

# Ensure all columns except 'Country Name' are numeric
for col in df_cleaned.columns:
    if col != "Country Name":
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Separate features (1960-2022) and target (2023 GDP)
X = df_cleaned.drop(columns=["2023", "Country Name"], errors='ignore')
y = df_cleaned["2023"]

# Check if there are still missing values
if X.isnull().sum().sum() > 0:
    print("Warning: Missing values still exist. Imputing...")

# Impute missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Ensure column names stay intact

# Ensure y has no missing values
y = y.fillna(y.mean())

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2e}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Ask the user for a country name
country_input = input("Enter a country name: ")

# Find the country data
country_row = df_cleaned[df_cleaned["Country Name"].str.lower() == country_input.lower()]

if not country_row.empty:
    # Get the country's GDP features (1960-2022)
    country_features = country_row.drop(columns=["2023", "Country Name"], errors='ignore')
    
    # Impute missing values in the country's data
    country_features_imputed = pd.DataFrame(imputer.transform(country_features), columns=country_features.columns)

    # Predict GDP for 2023
    predicted_gdp = model.predict(country_features_imputed)[0]
    
    print(f"Predicted GDP for {country_input} in 2023: ${predicted_gdp:,.2f} USD")

    # Plot actual vs. predicted GDP
    years = [int(year) for year in country_features.columns]  # Convert column names to integers
    actual_gdp = country_features.iloc[0].values  # Actual GDP from 1960-2022

    plt.figure(figsize=(12, 6))
    plt.plot(years, actual_gdp, marker='o', linestyle='-', label="Actual GDP (1960-2022)", color="blue")
    plt.scatter([2023], [predicted_gdp], color="red", label="Predicted GDP (2023)", s=100)

    # Add annotation for predicted GDP value
    plt.text(2023, predicted_gdp, f"${predicted_gdp:,.2f}", fontsize=12, color="red", ha="left", va="bottom")

    plt.xlabel("Year")
    plt.ylabel("GDP (in USD)")
    plt.title(f"GDP Trend for {country_input}: Actual (1960-2022) & Predicted (2023)")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("Country not found in the dataset.")
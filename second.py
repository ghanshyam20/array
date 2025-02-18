# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('50_Startups.csv')

# Check the first few rows
print("First 5 rows:")
print(df.head())

# List the variables
print("\nVariables:")
print(df.columns)

# Correlation heatmap for numerical columns
numerical_data = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Choose variables with high correlation to Profit
# R&D Spend and Marketing Spend are strongly correlated with Profit.

# Plot R&D Spend vs Profit
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['R&D Spend'], df['Profit'], color='blue', alpha=0.5)
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')

# Plot Marketing Spend vs Profit
plt.subplot(1, 2, 2)
plt.scatter(df['Marketing Spend'], df['Profit'], color='green', alpha=0.5)
plt.title('Marketing Spend vs Profit')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()

# Handle categorical 'State' column using One-Hot Encoding
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Check the dataset after encoding
print("\nDataset after encoding:")
print(df.head())

# Prepare features (X) and target (y)
X = df.drop(columns=['Profit'])
y = df['Profit']

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute RMSE and R2 for training data
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

# Compute RMSE and R2 for testing data
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# Print performance metrics
print("\nModel Performance:")
print(f"Training RMSE: {rmse_train}")
print(f"Training R2: {r2_train}")
print(f"Testing RMSE: {rmse_test}")
print(f"Testing R2: {r2_test}")

# Findings
"""
1. 'State' column was encoded using One-Hot Encoding.
2. 'R&D Spend' and 'Marketing Spend' are strong predictors of Profit.
3. Scatter plots show a linear relationship between these variables and Profit.
4. The model performs well with low RMSE and high R2 scores.
5. The model generalizes well to unseen data.
"""
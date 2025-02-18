




# we have imported the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# a) Which variable would you add next? Why?
# Let's add 'bp' (blood pressure) next because it is clinically relevant to diabetes progression.

# b) How does adding it affect the model's performance?
# We will compare the model's performance with just 'bmi' and 's5' versus adding 'bp'.

# Model with just 'bmi' and 's5'
X_bmi_s5 = X[['bmi', 's5']]
X_train, X_test, y_train, y_test = train_test_split(X_bmi_s5, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_bmi_s5 = np.sqrt(mean_squared_error(y_test, y_pred))
r2_bmi_s5 = r2_score(y_test, y_pred)

# Model with 'bmi', 's5', and 'bp'
X_bmi_s5_bp = X[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(X_bmi_s5_bp, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_bmi_s5_bp = np.sqrt(mean_squared_error(y_test, y_pred))
r2_bmi_s5_bp = r2_score(y_test, y_pred)

# d) Does it help if you add even more variables?
# Let's add all variables and see the performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred))
r2_all = r2_score(y_test, y_pred)

# Print metrics for comparison
print(f"RMSE with bmi and s5: {rmse_bmi_s5}")
print(f"R2 with bmi and s5: {r2_bmi_s5}")
print(f"RMSE with bmi, s5, and bp: {rmse_bmi_s5_bp}")
print(f"R2 with bmi, s5, and bp: {r2_bmi_s5_bp}")
print(f"RMSE with all variables: {rmse_all}")
print(f"R2 with all variables: {r2_all}")

# Create a heatmap to visualize correlation between variables
# Add the target variable 'y' to the dataframe for correlation analysis
X['target'] = y

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Diabetes Dataset')
plt.show()

# Create a heatmap to compare RMSE and R2 scores for different models
# Prepare data for the heatmap
models = ['bmi + s5', 'bmi + s5 + bp', 'All Variables']
metrics = ['RMSE', 'R2']
scores = np.array([[rmse_bmi_s5, r2_bmi_s5],
                   [rmse_bmi_s5_bp, r2_bmi_s5_bp],
                   [rmse_all, r2_all]])

# Plot the heatmap for model performance
plt.figure(figsize=(8, 6))
sns.heatmap(scores, annot=True, cmap='viridis', fmt='.2f', xticklabels=metrics, yticklabels=models)
plt.title('Model Performance Comparison (RMSE and R2)')
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.show()


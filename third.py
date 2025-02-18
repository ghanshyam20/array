# Import libraries
import pandas as pd  # For working with data
import numpy as np   # For math operations
import matplotlib.pyplot as plt  # For making graphs
from sklearn.model_selection import train_test_split  # To split data
from sklearn.preprocessing import StandardScaler  # To scale data
from sklearn.linear_model import Ridge, Lasso  # For Ridge and LASSO models
from sklearn.metrics import r2_score  # To check model performance

# Load the data
df = pd.read_csv("auto.csv")  # Read the car data from a file

# Keep only useful columns
# We want to predict 'mpg', so we drop 'name' and 'origin' (they’re not helpful)
X = df.drop(columns=["mpg", "name", "origin"])  # Features (what we use to predict)
y = df["mpg"]  # Target (what we want to predict)

# Scale the data to make all features equally important
scaler = StandardScaler()  # Create a scaler
X_scaled = scaler.fit_transform(X)  # Fit and transform the data

# Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Test different alpha values (alpha controls how strict the model is)
alphas = np.logspace(-3, 2, 50)  # Create 50 alpha values between 0.001 and 100

# Store scores for Ridge and LASSO
ridge_scores = []
lasso_scores = []

# Try each alpha and see how well the models perform
for alpha in alphas:
    # Ridge Regression
    ridge = Ridge(alpha=alpha)  # Create Ridge model
    ridge.fit(X_train, y_train)  # Train the model
    ridge_pred = ridge.predict(X_test)  # Make predictions
    ridge_scores.append(r2_score(y_test, ridge_pred))  # Save the score

    # LASSO Regression
    lasso = Lasso(alpha=alpha, max_iter=5000)  # Create LASSO model (allow more iterations)
    lasso.fit(X_train, y_train)  # Train the model
    lasso_pred = lasso.predict(X_test)  # Make predictions
    lasso_scores.append(r2_score(y_test, lasso_pred))  # Save the score

# Plot the scores for Ridge and LASSO
plt.figure(figsize=(10, 5))  # Create a graph
plt.plot(alphas, ridge_scores, label="Ridge R²", marker="o")  # Plot Ridge scores
plt.plot(alphas, lasso_scores, label="LASSO R²", marker="s")  # Plot LASSO scores
plt.xscale("log")  # Use log scale for alpha (easier to see)
plt.xlabel("Alpha (log scale)")  # Label for x-axis
plt.ylabel("R² Score")  # Label for y-axis
plt.title("R² Score vs Alpha for Ridge and LASSO")  # Title of the graph
plt.legend()  # Show labels
plt.grid(True)  # Add grid lines
plt.show()  # Display the graph

# Find the best alpha for Ridge and LASSO
best_ridge_alpha = alphas[np.argmax(ridge_scores)]  # Best alpha for Ridge
best_lasso_alpha = alphas[np.argmax(lasso_scores)]  # Best alpha for LASSO

# Print the best alpha values
print(f"Best Ridge Alpha: {best_ridge_alpha}")
print(f"Best LASSO Alpha: {best_lasso_alpha}")


"""
1) Best Alpha for Ridge and LASSO:
   FOR ridge the best alpha is around {optimal_ridge_alpha} and the R2 score is {optimal_ridge_r2}.
    FOR LASSO the best alpha is around {optimal_lasso_alpha} and the R2 score is {optimal_lasso_r2}.
2)Ridge works Better:
Ridge regression generally performs better than LASSO for this dataset.This means Ridge predicts the car's MPG more accurately as shown by its higher R^2 scores
.

3) Too Much Alpha Hurts:
When alpha becomes too large both Ridge and LASSO perform is bad.This happens because too much regularization makes the models too simple, and they can't capture the patterns in the data well.
4) Ridge is Less Sensitive: Ridge regression is more stable and less affected by chances in alpha . LASSO,on the other hand is more sensitive and its performance can drop quickly if alpha isnot choosen carefully.


,"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load CSV
df = pd.read_csv("linreg_data(1).csv")  # Replace with your CSV file path
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

# Fit and plot polynomial regression for degrees 2, 3, and 4
for d in [2, 3, 4]:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    plt.plot(X, model.predict(X_poly), label=f"Degree {d}")

plt.scatter(X, y, color="red")
plt.legend()
plt.show()





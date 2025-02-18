import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define x values and reshape it into a 2D array (required for sklearn)
x = np.linspace(-10, 10, 100).reshape(-1, 1)  # 100 points between -10 and 10
y = 2 * x.flatten() + 3  # y = 2x + 3 (a linear equation)

# Create the polynomial features for different degrees
for degree in [1, 2, 3, 4]:
    # Initialize PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    
    # Transform x to include polynomial terms (e.g., x^2, x^3, ...)
    x_poly = poly.fit_transform(x)
    
    # Fit the linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(x_poly, y)
    
    # Predict y values using the trained model
    y_poly_pred = model.predict(x_poly)
    
    # Plot the polynomial regression curve
    plt.plot(x, y_poly_pred, label=f"Degree {degree}", linewidth=2)

# Adding labels and title
plt.scatter(x, y, color='red', label="Original Data")  # Scatter the original data
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression with Different Degrees')
plt.legend()
plt.grid(True)
plt.show()

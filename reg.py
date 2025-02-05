import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We load the data from the file
data = pd.read_csv('linreg_data(1).csv', header=None, names=['X', 'Y'])

# Get the X (input) and Y (output) values from the data
input_values = data['X'].values
output_values = data['Y'].values
num_points = len(input_values)

# We need to calculate some numbers to draw the line
sum_x = np.sum(input_values)  # Add up all the X values
sum_y = np.sum(output_values)  # Add up all the Y values
sum_xy = np.sum(input_values * output_values)  # Add up X and Y multiplied together
sum_x_squared = np.sum(input_values**2)  # Add up X values squared

# Now, let's find out how steep the line should be and where it starts
slope = (num_points * sum_xy - sum_x * sum_y) / (num_points * sum_x_squared - sum_x**2)
intercept = (sum_y - slope * sum_x) / num_points

# Use the line we found to predict Y values from X
predicted_values = slope * input_values + intercept

# Find out how far off our predictions were (errors)
errors = output_values - predicted_values

# Let's see how big the errors are on average (Mean Squared Error)
mse = np.mean(errors**2)

# How well did our line fit the data? (R-squared)
mean_y = np.mean(output_values)  # Find the average of Y values
total_variance = np.sum((output_values - mean_y)**2)  # Measure how much Y values vary
error_variance = np.sum(errors**2)  # Measure how much the errors vary
r_squared = 1 - (error_variance / total_variance)

# Show the results (so we can see how our line did)
print(f"Slope of the line: {slope}")
print(f"Where the line starts: {intercept}")
print(f"How big the errors are (MSE): {mse}")
print(f"How well the line fits (R-squared): {r_squared}")

# Draw a picture with the points and the line
plt.scatter(input_values, output_values, color='red', label='Data Points')
plt.plot(input_values, predicted_values, color='blue', label='Regression Line')
plt.xlabel('Input (X)')
plt.ylabel('Output (Y)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

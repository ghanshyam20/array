import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'wt ht.csv'
data = pd.read_csv(file_path)

# Extract height and weight
heights_in_inches = data["Height"].values
weights_in_pounds = data["Weight"].values

# Convert heights to centimeters and weights to kilograms
heights_in_cm = heights_in_inches * 2.54
weights_in_kg = weights_in_pounds * 0.453592

# Calculate means
mean_height_cm = np.mean(heights_in_cm)
mean_weight_kg = np.mean(weights_in_kg)

# Print the mean values
print(f"Mean Height (cm): {mean_height_cm:.2f}")
print(f"Mean Weight (kg): {mean_weight_kg:.2f}")

# Plot histogram of heights in cm
plt.hist(heights_in_cm, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.show()

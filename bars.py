import matplotlib.pyplot as plt

# Step 1: Load the CSV file
import pandas as pd

file_path = 'wt ht.csv'  # Your file name
df = pd.read_csv(file_path)

# Step 2: Extract height and weight columns
height = df["Height"].tolist()
weight = df["Weight"].tolist()

# Step 3: Min-Max Normalization (Scaling values between 0 and 1)
min_height, max_height = min(height), max(height)
min_weight, max_weight = min(weight), max(weight)

height_normalized = [(x - min_height) / (max_height - min_height) for x in height]
weight_normalized = [(x - min_weight) / (max_weight - min_weight) for x in weight]

# Step 4: Z-score Standardization (Mean = 0, Std Dev = 1)
mean_height = sum(height) / len(height)
std_height = (sum((x - mean_height) ** 2 for x in height) / len(height)) ** 0.5

mean_weight = sum(weight) / len(weight)
std_weight = (sum((x - mean_weight) ** 2 for x in weight) / len(weight)) ** 0.5

height_standardized = [(x - mean_height) / std_height for x in height]
weight_standardized = [(x - mean_weight) / std_weight for x in weight]

# Step 5: Create a new DataFrame to store results
df_results = pd.DataFrame({
    "Original Height": height,
    "Normalized Height": height_normalized,
    "Standardized Height": height_standardized,
    "Original Weight": weight,
    "Normalized Weight": weight_normalized,
    "Standardized Weight": weight_standardized
})

# Step 6: Plot Histograms to visualize the data

plt.figure(figsize=(15, 5))

# Histogram for Original Data
plt.subplot(1, 3, 1)
plt.hist(df_results["Original Height"], bins=30, alpha=0.7, label="Height", color="blue")
plt.hist(df_results["Original Weight"], bins=30, alpha=0.7, label="Weight", color="orange")
plt.title("Original Data")
plt.legend()

# Histogram for Normalized Data
plt.subplot(1, 3, 2)
plt.hist(df_results["Normalized Height"], bins=30, alpha=0.7, label="Height", color="blue")
plt.hist(df_results["Normalized Weight"], bins=30, alpha=0.7, label="Weight", color="orange")
plt.title("Normalized Data (Min-Max)")
plt.legend()

# Histogram for Standardized Data
plt.subplot(1, 3, 3)
plt.hist(df_results["Standardized Height"], bins=30, alpha=0.7, label="Height", color="blue")
plt.hist(df_results["Standardized Weight"], bins=30, alpha=0.7, label="Weight", color="orange")
plt.title("Standardized Data (Z-score)")
plt.legend()

# Display all histograms
plt.tight_layout()
plt.show()

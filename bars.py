import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
data = pd.read_csv('Admission_Predict.csv')

# Check column names
print("Columns in the dataset:", data.columns)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check column names again
print("Columns after stripping spaces:", data.columns)

# Ensure the required columns exist
required_columns = ['Chance of Admit', 'Serial No.']
if not all(col in data.columns for col in required_columns):
    raise KeyError(f"Required columns {required_columns} not found in dataset.")

# Features and target variable
X = data.drop(columns=['Chance of Admit', 'Serial No.'])
y = data['Chance of Admit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model for normal data
model_normal = LinearRegression()
model_normal.fit(X_train, y_train)
y_pred_normal = model_normal.predict(X_test)
r2_normal = r2_score(y_test, y_pred_normal)

# Normalize the data (Min-Max scaling)
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Model for normalized data
model_minmax = LinearRegression()
model_minmax.fit(X_train_minmax, y_train)
y_pred_minmax = model_minmax.predict(X_test_minmax)
r2_minmax = r2_score(y_test, y_pred_minmax)

# Standardize the data (Z-score standardization)
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Model for standardized data
model_standard = LinearRegression()
model_standard.fit(X_train_standard, y_train)
y_pred_standard = model_standard.predict(X_test_standard)
r2_standard = r2_score(y_test, y_pred_standard)

# Print the R² scores
print(f"R² score for normal data: {r2_normal}")
print(f"R² score for normalized data: {r2_minmax}")
print(f"R² score for standardized data: {r2_standard}")
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# I loaded this really cool dataset called 'Iris', with 150 flower samples!
iris = datasets.load_iris()
X = iris.data  # This has all the features, like petal length and width
y = iris.target  # These are the flower types (Setosa, Versicolor, Virginica)

# I split the data into training and testing sets, so the model can learn and then test how well it learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, I had to scale the data (because SVM likes it when everything is on the same scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Just transform the test data, not fit

# Here comes the cool part: I created a Support Vector Machine (SVM) model using the RBF kernel. It's awesome for complex data!
svm = SVC(kernel='rbf')  # RBF kernel is like magic for this kind of task
svm.fit(X_train, y_train)  # I trained the model with the training data

# Time to test if the model is any good!
y_pred = svm.predict(X_test)  # The model makes predictions on the test data

# Look at what the model predicted and compare it with the actual labels!
print("Predictions:", y_pred)  # This is the model's guess
print("Actual Labels:", y_test)  # This is what the real labels were

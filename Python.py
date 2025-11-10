# Practical 1 : Design a simple machine learning model to train and test data

# Import the library to generate random numbers
from random import randint

# Import Linear Regression model from sklearn
from sklearn.linear_model import LinearRegression

# Empty lists to store training input (X) and output (y)
TRAIN_INPUT = []
TRAIN_OUTPUT = []

# Generate 100 random training examples
for i in range(100):
    # Create three random numbers as input features
    a = randint(0, 1000)
    b = randint(0, 1000)
    c = randint(0, 1000)

    # Generate output using a known formula: y = a + 2b + 3c
    op = a + (2 * b) + (3 * c)

    # Add input features to the input list
    TRAIN_INPUT.append([a, b, c])

    # Add the output value to the output list
    TRAIN_OUTPUT.append(op)

# Initialize the Linear Regression model
# n_jobs = -1 means use all CPU cores for faster training
predictor = LinearRegression(n_jobs=-1)

# Train (fit) the model using our training data
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

# Create test data (new input values)
X_TEST = [[10, 20, 30]]

# Predict the output for test data using the trained model
outcome = predictor.predict(X=X_TEST)

# Get the learned coefficients (weights for a, b, and c)
coefficients = predictor.coef_

# Display the prediction and coefficients
print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))

# Practical 2: Perform Data Loading and Feature Selection using PCA (Principal Component Analysis)

# Import required libraries
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA      # for Principal Component Analysis
from sklearn.feature_selection import RFE   # for feature elimination (not used here but imported)
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Step 1: Load the dataset
# -----------------------------

# The CSV file contains data for diabetes prediction (Pima Indians dataset)
url = "pima-indians-diabetes.csv"

# Define column names (based on the dataset structure)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Read the CSV file into a pandas DataFrame
dataframe = read_csv(url, names=names)

# Convert DataFrame into a NumPy array for processing
array = dataframe.values

# Split the data into input (X) and output (Y)
# X = first 8 columns → features (independent variables)
# Y = last column → class label (dependent variable)
X = array[:, 0:8]
Y = array[:, 8]

# -----------------------------
# Step 2: Apply Principal Component Analysis (PCA)
# -----------------------------

# Create a PCA model that will reduce 8 features → 3 principal components
pca = PCA(n_components=3)

# Fit the PCA model on input features (finds directions of maximum variance)
fit = pca.fit(X)

# -----------------------------
# Step 3: Display the results
# -----------------------------

# Show how much variance each of the 3 components explains
print("Explained Variance: %s" % fit.explained_variance_ratio_)

# Show the actual components (new feature combinations)
print(fit.components_)

# Practical 3: Perform Data Loading, Feature Selection, Feature Scoring, and Ranking

# Import required libraries
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier   # for finding feature importance

# ---------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------

# Dataset: Pima Indians Diabetes Dataset (used for classification)
url = "pima-indians-diabetes.csv"

# Define column names for the dataset
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Read the CSV file into a pandas DataFrame
dataframe = read_csv(url, names=names)

# Convert DataFrame into a NumPy array for processing
array = dataframe.values

# Separate the input features (X) and the output label (Y)
# X → first 8 columns (independent variables)
# Y → last column (dependent variable)
X = array[:, 0:8]
Y = array[:, 8]

# ---------------------------------------------------
# Step 2: Feature Extraction and Scoring
# ---------------------------------------------------

# Create the Extra Trees Classifier model
# n_estimators = 10 means the model will use 10 decision trees
model = ExtraTreesClassifier(n_estimators=10)

# Train (fit) the model on the dataset
model.fit(X, Y)

# ---------------------------------------------------
# Step 3: Display Feature Importance
# ---------------------------------------------------

# Print feature names (for reference)
print("Feature Names:")
print(names)

# Print importance scores for each feature
print("\nFeature Importance Scores:")
print(model.feature_importances_)

# Practical 4: Write a program to implement Decision Tree

# Import required libraries
from matplotlib import pyplot as plt              # For plotting the decision tree
from sklearn import datasets                      # For loading built-in datasets
from sklearn.tree import DecisionTreeClassifier   # For creating Decision Tree model
from sklearn import tree                          # For visualizing the tree

# ---------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------

# Load the Iris dataset (a built-in dataset in sklearn)
# It contains data of 3 flower types with 4 input features
iris = datasets.load_iris()

# X = input features (sepal length, sepal width, petal length, petal width)
X = iris.data

# y = output labels (flower species)
y = iris.target

# ---------------------------------------------------
# Step 2: Create and Train the Decision Tree model
# ---------------------------------------------------

# Create a Decision Tree Classifier model
# random_state=1234 ensures same result every time (for reproducibility)
clf = DecisionTreeClassifier(random_state=1234)

# Train (fit) the model on the dataset
model = clf.fit(X, y)

# ---------------------------------------------------
# Step 3: Visualize the Decision Tree
# ---------------------------------------------------

# Create a new figure for plotting
fig = plt.figure()

# Plot the trained Decision Tree with feature and class names
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,   # Column names for features
                   class_names=iris.target_names,      # Flower species names
                   filled=True)                        # Color the boxes for clarity

# Display the plotted Decision Tree
plt.show()


# Practical 5: Implement the Naïve Bayesian Classifier for a sample training dataset (CSV file)

# ---------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------
import pandas as pd                                      # For reading and handling CSV data
from sklearn.model_selection import train_test_split     # For splitting data into training and testing sets
from sklearn.naive_bayes import CategoricalNB            # Naïve Bayes classifier for categorical data
from sklearn.preprocessing import LabelEncoder           # For converting text labels to numeric form
from sklearn.metrics import accuracy_score               # To measure the accuracy of the model

# ---------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------
# Load data from CSV file (example: weather.csv)
data = pd.read_csv("weather.csv")

# Display the dataset
print("Training Data:\n", data, "\n")

# ---------------------------------------------------
# Step 3: Separate features and target
# ---------------------------------------------------
# X = input features (all columns except last)
# y = target/output column (the last column)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ---------------------------------------------------
# Step 4: Encode categorical values into numeric form
# ---------------------------------------------------
# Naïve Bayes works with numeric data, so we convert categories (like 'Sunny', 'Rainy') to numbers
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)   # Apply encoding to all columns
y_encoded = le.fit_transform(y)         # Encode target labels

# ---------------------------------------------------
# Step 5: Split the dataset into Training and Testing sets
# ---------------------------------------------------
# test_size=0.3 means 30% of the data is used for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3)

# ---------------------------------------------------
# Step 6: Train the Naïve Bayes model
# ---------------------------------------------------
model = CategoricalNB()          # Create Naïve Bayes model (for categorical data)
model.fit(X_train, y_train)      # Train (fit) the model on training data

# ---------------------------------------------------
# Step 7: Make predictions on test data
# ---------------------------------------------------
y_pred = model.predict(X_test)   # Predict the output for test data

# ---------------------------------------------------
# Step 8: Evaluate model performance
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)   # Compare predictions with actual labels

# ---------------------------------------------------
# Step 9: Display results
# ---------------------------------------------------
print("Test Data:\n", X_test, "\n")
print("Actual Labels:", y_test.tolist())
print("Predicted Labels:", y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# Practical 6: Implement Linear Regression algorithm using data from a CSV file

# ---------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------
import numpy as np                 # For numerical calculations
import matplotlib.pyplot as plt    # For plotting graphs
import pandas as pd                # For loading and handling CSV data

# ---------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------
# Read data from 'data.csv' file
datas = pd.read_csv('C:/Users/prasad/Downloads/data1.csv')

# Display the dataset
print("Dataset:\n", datas)

# ---------------------------------------------------
# Step 3: Divide dataset into independent and dependent variables
# ---------------------------------------------------
# iloc[:, 1:2] selects the 2nd column (input feature) → X
# iloc[:, 2] selects the 3rd column (output variable) → y
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# ---------------------------------------------------
# Step 4: Train (Fit) the Linear Regression model
# ---------------------------------------------------
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
lin = LinearRegression()

# Fit the model using the training data (X and y)
lin.fit(X, y)

# ---------------------------------------------------
# Step 5: Visualize the Linear Regression line
# ---------------------------------------------------
# Scatter plot for actual data points
plt.scatter(X, y, color='blue', label='Actual Data')

# Plot the best-fit regression line predicted by the model
plt.plot(X, lin.predict(X), color='red', label='Regression Line')

# Add chart title and labels
plt.title('Linear Regression')
plt.xlabel('Temperature')   # Independent variable (example)
plt.ylabel('Pressure')      # Dependent variable (example)
plt.legend()

# Display the plot
plt.show()


# Practical 7: Implement k-Nearest Neighbour (k-NN) algorithm to classify the Iris dataset

# ---------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------
import pandas as pd                                # For handling tabular data
import matplotlib.pyplot as plt                    # For visualization (optional)
from sklearn.datasets import load_iris             # To load the Iris dataset
from sklearn.model_selection import train_test_split  # For splitting train/test data
from sklearn.neighbors import KNeighborsClassifier    # KNN model

# ---------------------------------------------------
# Step 2: Load the Iris dataset
# ---------------------------------------------------
iris = load_iris()

# Create a pandas DataFrame from the dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display the features
print("Iris Dataset Features:\n", df)

# Add target column (numerical class values)
df['target'] = iris.target

# Add flower names (labels) using target mapping
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Display complete dataset with class names
print("\nComplete Dataset with Target Names:\n", df)

# ---------------------------------------------------
# Step 3: Separate each flower species (optional visualization step)
# ---------------------------------------------------
df0 = df[:50]    # Setosa
df1 = df[50:100] # Versicolor
df2 = df[100:]   # Virginica

# ---------------------------------------------------
# Step 4: Split dataset into features (X) and labels (y)
# ---------------------------------------------------
X = df.drop(['target', 'flower_name'], axis='columns')  # Input features
y = df.target                                           # Output labels

# ---------------------------------------------------
# Step 5: Split data into training and testing sets
# ---------------------------------------------------
# test_size=0.2 → 20% data for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------------------------------
# Step 6: Create and train the KNN classifier
# ---------------------------------------------------
# n_neighbors=10 → looks at 10 nearest data points for prediction
knn = KNeighborsClassifier(n_neighbors=10)

# Fit the KNN model using training data
knn.fit(X_train, y_train)

# ---------------------------------------------------
# Step 7: Test the model and display accuracy
# ---------------------------------------------------
# Evaluate model accuracy on test data
accuracy = knn.score(X_test, y_test)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))


# Practical 8: Implement Classification Model using K-Means Clustering

# ---------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------
# Step 2: Create sample data
# ---------------------------------------------------
# Example data points (x, y)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combine x and y into one dataset (list of coordinate pairs)
data = list(zip(x, y))

# ---------------------------------------------------
# Step 3: Find the optimal number of clusters using the Elbow Method
# ---------------------------------------------------
inertias = []  # List to store inertia values for each k

for i in range(1, 11):  # Try 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)  # Store the sum of squared distances

# Plot the Elbow graph
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.grid(True)
plt.show()

# ---------------------------------------------------
# Step 4: Apply K-Means clustering with the chosen number of clusters (k=2)
# ---------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

# ---------------------------------------------------
# Step 5: Visualize the clustered data
# ---------------------------------------------------
plt.scatter(x, y, c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='black', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering (k=2)')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.show()


# Practical 9: Implement Classification Model using Hierarchical Clustering (with Prediction)

# ---------------------------------------------------
# Step 1: Import necessary libraries
# ---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------------------
# Step 2: Create the dataset
# ---------------------------------------------------
# Example data points (X, Y)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combine x and y coordinates into a list of pairs
data = list(zip(x, y))

# ---------------------------------------------------
# Step 3: Perform Hierarchical Clustering
# ---------------------------------------------------
# AgglomerativeClustering = bottom-up approach
# n_clusters = desired number of groups
# metric = distance measure (Euclidean)
# linkage = method to minimize variance between clusters (Ward’s method)
hierarchical_cluster = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)

# Fit the model and predict cluster labels for each point
labels = hierarchical_cluster.fit_predict(data)

# ---------------------------------------------------
# Step 4: Print cluster labels for each data point
# ---------------------------------------------------
print("Cluster Labels for each data point:")
print(labels)

# ---------------------------------------------------
# Step 5: Plot the clustered data
# ---------------------------------------------------
plt.scatter(x, y, c=labels, cmap='rainbow')
plt.title('Hierarchical Clustering (Agglomerative)')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.show()

# ---------------------------------------------------
# Step 6 (Optional): Display Dendrogram for visual explanation
# ---------------------------------------------------
# The dendrogram shows how clusters are formed step by step.
linked = linkage(data, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

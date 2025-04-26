#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries for data manipulation and visualization
import pandas as pd  # For data handling and manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced data visualization

# Importing scikit-learn modules for machine learning operations
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler  # For feature scaling (standardization)
from sklearn.decomposition import PCA  # For Principal Component Analysis (dimensionality reduction)
from sklearn.ensemble import RandomForestClassifier  # For building a Random Forest classification model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation metrics
from sklearn.manifold import TSNE  # For data visualization using t-Distributed Stochastic Neighbor Embedding
from sklearn.impute import SimpleImputer  # For handling missing values by imputing them


# Loading the dataset from a CSV file into a pandas DataFrame
df = pd.read_csv('data (2).csv')


# Removing any columns that have names starting with 'Unnamed'
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Mapping the 'diagnosis' column to a new 'target' column (B -> 0, M -> 1)
df['target'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Dropping the 'id' and original 'diagnosis' columns from the DataFrame
df = df.drop(['id', 'diagnosis'], axis=1)

# Separating the features (X) and the target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an imputer object to fill missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')

# Applying the imputer to the training set and converting it back to a DataFrame
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)

# Applying the same transformation to the testing set
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Creating a scaler object to standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fitting the scaler on the training set and transforming it
X_train_scaled = scaler.fit_transform(X_train)

# Transforming the testing set using the same scaler fitted on the training set
X_test_scaled = scaler.transform(X_test)

# Creating a PCA object to perform principal component analysis (PCA) for dimensionality reduction
pca_full = PCA()

# Fitting the PCA model on the scaled training data
pca_full.fit(X_train_scaled)

# Calculating the cumulative explained variance ratio from the PCA model
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plotting the cumulative variance as a line graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='darkblue')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% explained variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Finding the number of components required to explain at least 95% of the variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components required to retain 95% of the variance: {n_components}")

# Performing PCA with the number of components that explain at least 95% of the variance
pca = PCA(n_components=n_components)

# Transforming the scaled training data using the fitted PCA model
X_train_pca = pca.fit_transform(X_train_scaled)

# Transforming the scaled testing data using the same PCA model
X_test_pca = pca.transform(X_test_scaled)

# Initializing a Random Forest Classifier model with a fixed random state for reproducibility
model = RandomForestClassifier(random_state=42)

# Fitting the model to the training data (using PCA-transformed features)
model.fit(X_train_pca, y_train)

# Making predictions on the test set (using PCA-transformed features)
y_pred = model.predict(X_test_pca)

# Calculating and printing the accuracy score of the model's predictions on the test data
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Creating a heatmap for the confusion matrix to visualize model performance
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Mapping the predicted labels (0 or 1) to their corresponding disease types
disease_types = {0: "Benign (B)", 1: "Malignant (M)"}

# Creating a list of predicted disease types using the mapped labels
predicted_disease_type = [disease_types[label] for label in y_pred]

# Printing the predicted disease types for the first 10 test samples
print("\nDisease Types (for each sample):")
for i, disease in enumerate(predicted_disease_type[:10]):  # Displaying the first 10 samples
    print(f"Test Sample {i+1}: {disease}")

# Performing PCA with 10 components for t-SNE visualization
pca_for_tsne = PCA(n_components=10)
X_train_pca10 = pca_for_tsne.fit_transform(X_train_scaled)

# Initializing t-SNE with 2 components to reduce dimensionality for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, learning_rate='auto', init='pca', random_state=42)

# Transforming the PCA-reduced data using t-SNE for 2D visualization
X_tsne = tsne.fit_transform(X_train_pca10)

# Creating a scatter plot to visualize the 2D t-SNE projection
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[y_train == 0, 0], X_tsne[y_train == 0, 1], label='Benign (0)', alpha=0.7, s=60, c='green')
plt.scatter(X_tsne[y_train == 1, 0], X_tsne[y_train == 1, 1], label='Malignant (1)', alpha=0.7, s=60, c='crimson')
plt.title('2D Visualization using T-SNE (by Disease Status)')
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

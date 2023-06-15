import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json
import os
# Load data from CSV file
data = pd.read_csv('data/covtype.csv', header=None)

# Split data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier on the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict labels for the test data
y_pred = clf.predict(X_test)

# Calculate accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)


print(f'Accuracy: {accuracy:.4f}')


# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": accuracy}, outfile)
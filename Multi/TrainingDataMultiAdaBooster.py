import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# Load the CSV file
training_data = pd.read_csv('TrainingDataMulti.csv')

# Split the dataset into features and labels
X_train = training_data.drop('marker', axis=1)
y_train = training_data['marker']

# Define the parameter grid to search over
param_grid = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.1, 0.5, 1.0]}

# Create an AdaBoost classifier
clf = AdaBoostClassifier(random_state=42)

# Use grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Use the best hyperparameters to train the classifier
clf = AdaBoostClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                         learning_rate=grid_search.best_params_['learning_rate'],
                         random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = clf.predict(X_train)

# Calculate the accuracy of the classifier on the training data
train_acc = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: {:.2f}%".format(train_acc*100))

# Load the testing CSV file
test_data = pd.read_csv('TestingDataMulti.csv')

# Predict the labels for the testing data
y_test_pred = clf.predict(test_data)

# Add the predicted labels to the testing data
test_data['marker'] = y_test_pred

# Save the results to a CSV file
test_data.to_csv('TestingResultsMulti.csv', index=False)

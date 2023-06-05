import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from wittgenstein import JRip


# Load the CSV file
data = pd.read_csv('TrainingDataBinary.csv')

# Split the dataset into features and labels
X = data.drop('marker', axis=1)
y = data['marker']

# Create a pipeline with CountVectorizer and JRip
pipe = make_pipeline(
    CountVectorizer(),
    JRip()
)

# Create an AdaBoost classifier with 100 estimators and a learning rate of 1.0
clf = AdaBoostClassifier(base_estimator=pipe, n_estimators=100, learning_rate=1.0, random_state=42)

# Use cross-validation to evaluate the classifier
scores = cross_val_score(clf, X, y, cv=5)

# Print the average accuracy across all folds
print("Average accuracy: {:.2f}%".format(scores.mean()*100))

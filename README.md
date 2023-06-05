Detection of Attacks on Power System Grids
This repository contains the code and report for the coursework on the detection of attacks on power system grids. The project is divided into two parts: Part A focuses on binary classification, while Part B involves multi-class classification.

PART A: Binary Classification
Introduction
The purpose of Part A is to design and implement a machine learning model that can accurately classify system traces as normal events or abnormal data injection attack events. The dataset provided consists of 6,000 system traces, each described by 128 features. Half of the traces represent normal events, while the other half represents abnormal data injection attacks.

Dataset
The dataset is provided in the Binary folder and includes the following files:

TrainingDataBinary.csv: The training dataset containing labeled system traces.
TestingDataBinary.csv: The testing dataset containing unlabeled system traces.
Methodology
In Part A, an AdaBoost classifier is used as the machine learning model. The AdaBoost algorithm combines multiple weak classifiers to create a strong classifier. The implementation includes the following steps:

Loading and preprocessing the training data.
Defining a grid of hyperparameters for the AdaBoost classifier.
Performing grid search with cross-validation to find the best hyperparameters.
Training the AdaBoost classifier with the best hyperparameters.
Making predictions on the training data and calculating the accuracy of the classifier.
Loading and preprocessing the testing data.
Using the trained classifier to predict the labels for the testing data.
Saving the predicted labels in the TestingResultsBinary.csv file.
Results
The trained AdaBoost classifier achieved an accuracy of 90.88% on the training data. The predictions for the testing data are saved in the TestingResultsBinary.csv file.

PART B: Multi-Class Classification
Introduction
Part B focuses on multi-class classification, where the goal is to predict the types of events in the system traces. The dataset includes three types of events: normal events, abnormal data injection attacks, and abnormal command injection attacks.

Dataset
The dataset for Part B is located in the Multi folder and includes the following files:

TrainingDataMulti.csv: The training dataset containing labeled system traces with multiple event types.
TestingDataMulti.csv: The testing dataset containing unlabeled system traces.
Methodology
In Part B, an AdaBoost classifier is also used as the machine learning model. The implementation follows similar steps to Part A, including loading the data, defining hyperparameters, performing grid search, training the classifier, making predictions, and saving the results in the TestingResultsMulti.csv file.

Results
The training accuracy achieved for the multi-class classification task is 73.68%. The predictions for the testing data are saved in the TestingResultsMulti.csv file.

Repository Contents
Binary folder: Contains the code and files related to binary classification.
Multi folder: Contains the code and files related to multi-class classification.
README.md: Provides an overview of the project, including the problem, methodology, and results.
Please refer to the specific folders for the code implementation of each part.

Report and Analysis
The detailed report for this project, including an analysis of the results.

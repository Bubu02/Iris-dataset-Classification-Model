# Iris Classification Model

This repository contains a machine learning model for classifying Iris species. The model was trained on the famous Iris dataset and evaluated using various metrics such as accuracy, precision, recall, and F1 score. A precision-recall curve was also plotted to visualize the performance of the model.

## Project Structure

- `Iris_Classification.ipynb`: This Jupyter notebook contains all the code for data loading, preprocessing, model training, evaluation, and visualization.
- `Iris_classification_svm.pkl`: This is the trained Support Vector Machine (SVM) model saved in pickle format.

## Model

The task of classifying Iris species was approached by training various models including SVM, Logistic Regression, and Random Forest. Each model's hyperparameters were tuned using GridSearchCV. After comparing the performance of these models, SVM was found to be the best and was chosen for the final model.

## Evaluation

The model was evaluated using various metrics:
- Accuracy: This is the ratio of correct predictions to total predictions.
- Precision: This is the ratio of true positives to the sum of true positives and false positives.
- Recall: This is the ratio of true positives to the sum of true positives and false negatives.
- F1 Score: This is the harmonic mean of precision and recall.

A precision-recall curve was also plotted to visualize these metrics.

## Dependencies

This project requires Python and the following Python libraries installed:
- NumPy
- Pandas
- scikit-learn
- Matplotlib


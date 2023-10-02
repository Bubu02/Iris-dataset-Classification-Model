from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import joblib

iris = datasets.load_iris()
# print(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
# print(df[47:52])

# Train-Test splitting
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C' : [1,10,20],
            'kernel': ['rbf', 'linear']
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'C':[1,5,10]
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10]
        }
    }

}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model','best_score', 'best_params'])
print(df)

# Corrected hyperparameters
best_svm = svm.SVC(C=1, kernel='rbf', gamma='auto')
best_svm.fit(x_train, y_train)
y_pred = best_svm.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Iterate over each class
classes = [0, 1, 2]  # Assuming 0, 1, and 2 are the class labels
plt.figure(figsize=(8, 6))

for class_label in classes:
    # Create binary labels for the current class vs. all others
    y_test_binary = (y_test == class_label)

    # Calculate predicted probabilities for the positive class (class_label)
    y_scores = best_svm.decision_function(x_test)[:, class_label]

    # Compute precision and recall values
    precision, recall, _ = precision_recall_curve(y_test_binary, y_scores)

    # Plot the precision-recall curve for the current class
    plt.plot(recall, precision, label=f'Class {class_label}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.grid()
plt.legend()
# plt.show()

# Assuming you have 'best_svm' as your trained SVM model
Iris_Classification = "Iris_classification_svm.pkl"

# Save the model to a file
joblib.dump(best_svm, Iris_Classification)

print(f"Model saved as {Iris_Classification}")
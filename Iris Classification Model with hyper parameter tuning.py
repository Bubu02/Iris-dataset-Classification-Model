from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
# print(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
# print(df[47:52])

# Train-Test splitting
x_train, x_text, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C' : [1,10,20],
            'kernal': ['rbf', 'linear']
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
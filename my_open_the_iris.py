import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from pandas import read_csv
from pandas.plotting import scatter_matrix

def load_dataset():
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names, header=0)
    return dataset

def summarize_dataset(dataset):
    print("its shape:")
    print(dataset.shape)
    print()
    print("its ten first lines:")
    print(dataset.head(10))
    print()
    print("its statistical summary:")
    print(dataset.describe())
    print()
    print("Its distribution:")
    print(dataset.groupby('class').size())

dataset = load_dataset()
summarize_dataset(dataset)

def print_plot_univariate(dataset):
    dataset.hist()
    pyplot.show()

dataset = load_dataset()
print_plot_univariate(dataset)

def print_plot_multivariate(dataset):
    scatter_matrix(dataset)
    pyplot.show()
    
dataset = load_dataset()
print_plot_multivariate(dataset)

def my_print_and_test_models(dataset):
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    models = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('GaussianNB', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LinearDiscriminant', LinearDiscriminantAnalysis()),
        ('SVM', SVC(gamma='auto'))
    ]
     # evaluate each model
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

dataset = load_dataset()
my_print_and_test_models(dataset)

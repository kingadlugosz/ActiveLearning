import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss, \
    precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.datasets import load_wine, load_iris, fetch_20newsgroups
from sklearn.base import clone
import methods
import os
import pandas as pd

path = os.getcwd()

classifiers = [
    (SVC(kernel="linear", C=0.025, probability=True), 'SVC'),
    (KNeighborsClassifier(3), 'KNN'),
    (GaussianNB(), 'Naive Bayes'),
]

datasets = [
    (load_wine(return_X_y=True), 'wine'),
    (load_iris(return_X_y=True), 'iris'),
    # (fetch_20newsgroups(return_X_y=True), '20 newsgroups'),
]

labeling_methods = [
    (methods.random_selection, 'random'),
    (methods.maxi_mini_selection, 'maxi mini'),

]

if __name__ == '__main__':
    for dataset in datasets:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[0][0], dataset[0][1], test_size=0.4, random_state=777
        )

        for classifier in classifiers:
            for method in labeling_methods:
                x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
                    x_train, y_train, train_size=10, random_state=777
                )
                acc = []
                loss = []
                while x_unlabeled.shape[0] > 50:
                    model = clone(classifier[0])
                    model.fit(x_labeled, y_labeled)
                    y_pred = model.predict(x_test)

                    # y_unlabeled = classifier.predict(x_unlabeled)
                    x_labeled, x_unlabeled, y_labeled, y_unlabeled = method[0](model,x_labeled, x_unlabeled, y_labeled,
                                                                               y_unlabeled)
                    # plot_confusion_matrix(classifier, x_test, y_test, cmap='GnBu')
                    loss.append(log_loss(y_labeled,model.predict_proba(x_labeled)))
                    acc.append(accuracy_score(y_test, y_pred))
                plt.plot(loss)
                plt.title(f'classifier: {classifier[1]}, dataset: {dataset[1]}, labeling method: {method[1]}\nloss')
                plt.show()
                plt.plot(acc)
                plt.title(f'classifier: {classifier[1]}, dataset: {dataset[1]}, labeling method: {method[1]}\nacc')
                plt.show()

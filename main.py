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


class ActiveLearning:
    def __init__(self, dataset, classifier, labeling_method, test_size=0.4, start_labels=10, seed=777):
        self.title = f'classifier: {classifier[1]}, dataset: {dataset[1]}, labeling method: {method[1]}\n'

        x_train, self.x_test, oracle_train, self.y_test = train_test_split(
            dataset[0][0], dataset[0][1], test_size=test_size, random_state=seed
        )

        self.x_labeled, self.x_unlabeled, self.y_labeled, self.y_unlabeled = train_test_split(
            x_train, oracle_train, train_size=start_labels, random_state=seed
        )
        self.method = labeling_method[0]
        self.classifier = clone(classifier[0])
        self.accuracies = []
        self.losses = []

    def train(self, stop=50):
        while self.x_unlabeled.shape[0] > stop:
            self.classifier.fit(self.x_labeled, self.y_labeled)

            self.x_labeled, self.x_unlabeled, self.y_labeled, self.y_unlabeled = self.method(self.classifier,
                                                                                             self.x_labeled,
                                                                                             self.x_unlabeled,
                                                                                             self.y_labeled,
                                                                                             self.y_unlabeled)

            y_pred = self.classifier.predict(self.x_test)

            self.losses.append(log_loss(self.y_labeled, self.classifier.predict_proba(self.x_labeled)))
            self.accuracies.append(accuracy_score(self.y_test, y_pred))

    def plot(self):
        plt.plot(self.losses)
        plt.title(self.title + 'loss')
        plt.show()
        plt.plot(self.accuracies)
        plt.title(self.title + 'accuracy')
        plt.show()


if __name__ == '__main__':
    for dataset in datasets:
        for classifier in classifiers:
            for method in labeling_methods:
                model = ActiveLearning(dataset, classifier, method)
                model.train()
                model.plot()

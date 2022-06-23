import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss, \
    precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.datasets import load_wine, load_iris
from sklearn.base import clone
import methods
import os
from datasets import load_dataset
from copy import copy
from sklearn.model_selection import KFold

path = os.getcwd()

classifiers = [
    (SVC(kernel="linear", C=0.025, probability=True), 'SVC'),
    (KNeighborsClassifier(3), 'KNN'),
    (GaussianNB(), 'Naive Bayes'),
]

datasets = [
    (load_wine(return_X_y=True), 'wine'),
    (load_iris(return_X_y=True), 'iris'),
    (load_dataset('datasets/banana.csv'), 'banana'),
    (load_dataset('datasets/magic.csv'), 'magic'),
    (load_dataset('datasets/titanic.csv'), 'titanic'),
]

labeling_methods = [
    (methods.random_selection, 'random'),
    # (methods.maxi_mini_selection, 'maxi mini'),
    (methods.least_confidence, 'least confidence'),
    (methods.margin_sampling, 'margin sampling'),

]




class ActiveLearning:
    def __init__(self, dataset, classifier, labeling_method, train_idx, test_idx, test_size=0.6, start_labels=10, lr=1,
                 seed=777):
        # self.title = f'classifier: {classifier[1]}, dataset: {dataset[1]}, labeling method: {method[1]}\n'
        self.title = f'{method[1]}'
        self.lr = lr
        # self.x_train, self.x_test, self.oracle_train, self.y_test = train_test_split(
        #     dataset[0][0], dataset[0][1], test_size=test_size, random_state=seed
        # )
        self.x_train, self.x_test = dataset[0][0][train_idx], dataset[0][0][test_idx]
        self.oracle_train, self.y_test = dataset[0][1][train_idx], dataset[0][1][test_idx]
        self.x_labeled, self.x_unlabeled, self.y_labeled, self.y_unlabeled = train_test_split(
            self.x_train, self.oracle_train, train_size=start_labels, random_state=seed
        )
        self.method = labeling_method[0]
        self.classifier = clone(classifier[0])
        self.accuracies = []
        self.losses = []

    def train(self, stop=0.5):
        while self.x_unlabeled.shape[0] / self.x_train.shape[0] > stop:
            self.classifier.fit(self.x_labeled, self.y_labeled)
            self.x_labeled, self.x_unlabeled, self.y_labeled, self.y_unlabeled = self.method(self.classifier,
                                                                                             self.x_labeled,
                                                                                             self.x_unlabeled,
                                                                                             self.y_labeled,
                                                                                             self.y_unlabeled,
                                                                                             size=self.lr)

            y_pred = self.classifier.predict(self.x_test)

            self.losses.append(log_loss(self.oracle_train, self.classifier.predict_proba(self.x_train)))
            self.accuracies.append(accuracy_score(self.y_test, y_pred))

    def get_hist(self):
        return self.losses, self.accuracies, self.title

    def plot(self):
        # plt.plot(self.losses)
        # plt.title(self.title + 'loss')
        # plt.show()
        plt.figure()
        plt.plot(self.accuracies)
        plt.title(self.title + 'accuracy')
        plt.show()


if __name__ == '__main__':
    scores = {method[1]: [] for method in labeling_methods}
    n_splits = 5
    for dataset in datasets:
        for classifier in classifiers:
            all_acc = []
            all_loss = []
            methods = []
            for method in labeling_methods:
                method_acc = []
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=777)
                # tmp = dataset[0][0].shape[0] % n_splits
                # [:, -1 * tmp]
                for train_idx, test_idx in kf.split(dataset[0][0]):
                    model = ActiveLearning(dataset, classifier, method, train_idx, test_idx, lr=1, start_labels=10)
                    if dataset[1] == 'iris' or dataset[1] == 'wine':
                        stop = 0.5
                    elif dataset[1] == 'magic':
                        stop = 0.99
                    else:
                        stop = 0.95
                    model.train(stop=stop)
                    loss, acc, met = model.get_hist()
                    if len(method_acc):
                        acc = acc[:len(method_acc[0])]
                    method_acc.append(copy(acc))
                    all_loss.append(copy(loss))
                    # model.plot()
                np_method_acc = np.array(method_acc)
                scores[met].append(np.mean(np_method_acc[:, -1], axis=0))
                all_acc.append(np.mean(method_acc, axis=0))
                methods.append(met)
            for i in range(len(all_acc)):
                plt.plot(all_acc[i])
            plt.title(f'classifier: {classifier[1]}, dataset: {dataset[1]}')
            plt.legend(methods)
            plt.show()

            # for i in range(len(methods)):
            #     plt.plot(all_loss[i])
            # plt.title(f'classifier: {classifier[1]}, dataset: {dataset[1]}')
            # plt.legend(methods)
            # plt.show()
# scores = np.array(scores)

    np_scores = []
    for vector in scores.items():
        np_scores.append(vector[1])
    np_scores = np.array(np_scores)

    np.save('results', np_scores)

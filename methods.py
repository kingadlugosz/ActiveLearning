from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss


def random_selection(model, x_labeled, x_unlabeled, y_labeled, y_unlabeled, size=1):
    new_x_labeled, new_x_unlabeled, new_y_labeled, new_y_unlabeled = train_test_split(
        x_unlabeled, y_unlabeled, train_size=size
    )
    return np.concatenate((x_labeled, new_x_labeled)), new_x_unlabeled, np.concatenate(
        (y_labeled, new_y_labeled)), new_y_unlabeled


def maxi_mini_selection(model, x_labeled, x_unlabeled, y_labeled, y_unlabeled, size=1):
    maxi = 0
    pred = model.predict_proba(x_unlabeled[maxi].reshape(1, -1))
    labels = [i for i in range(pred.shape[1])]
    maxi_loss = log_loss([y_unlabeled[maxi]], pred, labels=labels)
    for i in range(x_unlabeled.shape[0]):
        pred = model.predict_proba(x_unlabeled[i].reshape(1, -1))
        loss = log_loss([y_unlabeled[i]], pred, labels=labels)
        if loss > maxi_loss:
            maxi_loss = loss
            maxi = i
    x_labeled = np.concatenate((x_labeled, x_unlabeled[maxi].reshape(1, -1)))
    y_labeled = np.concatenate((y_labeled, y_unlabeled[maxi].reshape(-1, )))
    x_unlabeled = np.delete(x_unlabeled, (maxi), axis=0)
    y_unlabeled = np.delete(y_unlabeled, (maxi))
    return x_labeled, x_unlabeled, y_labeled, y_unlabeled


def least_confidence(model, x_labeled, x_unlabeled, y_labeled, y_unlabeled, size=1):
    pred = model.predict_proba(x_unlabeled)
    pred = np.max(pred, axis=1, keepdims=True)
    for i in range(size):
        least = np.argmin(pred)
        pred[least] = 1

        x_labeled = np.concatenate((x_labeled, x_unlabeled[least].reshape(1, -1)))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[least].reshape(-1, )))
        x_unlabeled = np.delete(x_unlabeled, (least), axis=0)
        y_unlabeled = np.delete(y_unlabeled, (least))
    return x_labeled, x_unlabeled, y_labeled, y_unlabeled


def margin_sampling(model, x_labeled, x_unlabeled, y_labeled, y_unlabeled, size=1):
    pred = model.predict_proba(x_unlabeled)
    max = np.max(pred, axis=1, keepdims=True)
    second_max = np.max(pred, axis=1, where=pred < max, initial=-1, keepdims=True)
    margins = max - second_max
    for i in range(size):
        mini = np.argmin(margins)
        margins[mini] = 1

        x_labeled = np.concatenate((x_labeled, x_unlabeled[mini].reshape(1, -1)))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[mini].reshape(-1, )))
        x_unlabeled = np.delete(x_unlabeled, (mini), axis=0)
        y_unlabeled = np.delete(y_unlabeled, (mini))
    return x_labeled, x_unlabeled, y_labeled, y_unlabeled

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def sigmoid(inputs, weights, bias):
    """
    Applies the sigmoid function to the parameters of the perceptron
    :param inputs: inputs of the perceptron
    :param weights: weights of the perceptron
    :param bias: bias of the perceptron
    :return: output of the perceptron
    """
    z = np.dot(weights, inputs) + bias
    return 1 / (1 + np.exp(-z))


def step(inputs, weights, bias):
    """
    Applies the step function to the parameters of the perceptron
    :param inputs: inputs of the perceptron
    :param weights: weights of the perceptron
    :param bias: bias of the perceptron
    :return: output of the perceptron
    """
    return 1. if np.dot(weights, inputs) + bias > 0 else 0.


def tanh(inputs, weights, bias):
    """
    Applies the tanh function to the parameters of the perceptron
    :param inputs: inputs of the perceptron
    :param weights: weights of the perceptron
    :param bias: bias of the perceptron
    :return: output of the perceptron
    """
    return np.tanh(np.dot(weights, inputs) + bias)


# --------- FOR TRAINING - EPOCHS -------- #

#TO_BIN = lambda x : 0. if x<0.5 else 1.
#NP_TO_BIN = np.vectorize(TO_BIN)

#TO_BIN_TANH = lambda x : 0. if x<=0 else 1.
#NP_TO_BIN_TANH = np.vectorize(TO_BIN_TANH)


def epoch(NN, len_data, inputs, expected):
    """
    Computes an epoch for the neural network with the data and
    true classification of the data.
    This function trains the neural network.
    :param NN: NeuralNetwork
    :param len_data: number of examples
    :param inputs: training data
    :param expected: true classification of inputs
    :return: MSE of the epoch
    """
    assert len(inputs) == len(expected), "Mismatched lenghts"
    err = 0
    tot = 0
    for input, expect in zip(inputs, expected):
        err += NN.train(np.asarray(input), np.asarray(expect))
        tot += 1
    assert tot == len_data, "Mismatched lenghts of training inputs"
    return err / len_data


def hits(NN, inputs, expected, len_data):
    """
    Feeds the neural network and compares the output with the
    real classification of the data.
    :param NN: NeuralNetwork
    :param inputs: testing data
    :param expected: true classification of inputs
    :param len_data: number of examples
    :return: 1- Number of hits of the neural network
             2- Accuracy of the neural network
    """
    hit = 0
    tot = 0
    for input, expect in zip(inputs, expected):
        #got = NP_TO_BIN(NN.feed(np.asarray(input)))
        got = NN.get_last_activation().to_bin(NN.feed(np.asarray(input)))
        hit += np.array_equal(got, np.asarray(expect))
        tot += 1
    assert tot == len_data, "Mismatched lenghts of testing inputs"
    return hit, hit/len_data


# ------------- FOR DATASETS ------------- #

def get_classification(dataset):
    return dataset[:, -1]


def get_attributes(dataset):
    return dataset[:, :-1]


def normalize_dataset(dataset, min=0, max=1):
    """
    Normalizes the data in a dataset.
    :param dataset: dataset with values to normalize
    :param min: min value expected in the data
    :param max: max value expected in the data
    :return: the normalized dataset
    """
    scale = (max - min) / (dataset.max(axis=0) - dataset.min(axis=0))
    return scale * dataset + min - dataset.min(axis=0) * scale


def one_hot_encoding(u):
    """
    Given an array with the uniques classifications of the data,
    computes the 1-hot encoding.
    :param u: array with the uniques classifications
    :return: dict with the classification as key and code as value
    """
    hots = {}
    n = len(u)
    zeros = np.zeros(n)
    for i in range(n):
        zeros[i] = 1
        hots[u[i]] = np.array(zeros)
        zeros[i] = 0
    return hots


def hot_encode(classification):
    """
    Given the classification of the data, computes the
    1-hot encoding and encodes the data.
    :param classification: 1d array with
    :return: 1- the classification 1-hot encoded
             2- the dict with the codes
    """
    u = np.unique(classification)
    n = len(u)
    hots = one_hot_encoding(u)
    l = len(classification)
    ret = np.empty((l, n))
    for i in range(l):
        ret[i] = hots[classification[i]]
    return ret, hots


def confusion_matrix_(NN, hots, true, pred, plot=False):
    """
    Calculates (and plots) the confusion matrix for the given
    true and predicted data.
    :param hots: dict with classes and encodings
    :param true: true classification of the data
    :param pred: predicted classification of the data
    :param plot: boolean, True to plot the confusion matrix
    :return: confusion matrix
    """
    assert len(true) == len(pred), "Mismatched lengths"

    y_true = []
    y_pred = []
    for t, p in zip(true, pred):
        c_t = get_class(t, hots)
        c_p = get_class(NN.get_last_activation().to_bin(p), hots)
        if c_t and c_p:
            y_true.append(c_t)
            y_pred.append(c_p)

    labels = []
    for key in hots.keys():
        labels.append(str(key))

    if plot:
        _, cm = plot_confusion_matrix(y_true, y_pred,
                                      labels,
                                      title='Matriz de confusion para Datos de Testing')
        plt.show()
    else:
        cm = confusion_matrix(y_true, y_pred)

    return cm


def get_class(code, hots):
    """
    Given a code and the encoding dict, returns the true
    classification of the code.
    :param code: code (1-hot encoding)
    :param hots: dict with classes and encodings
    :return: the class if its a valid code, False otherwise
    """
    for key in hots.keys():
        if np.array_equal(hots[key], code):
            return key
    return False


def plot_confusion_matrix(y_true, y_pred, classes, title,
                          cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks
    ax.set(xticks=np.arange(start=0, stop=cm.shape[1]),
           yticks=np.arange(start=-0.5, stop=cm.shape[0]+0.5),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm
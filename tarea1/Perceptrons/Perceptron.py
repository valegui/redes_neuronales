import numpy as np
from utils import *


class Perceptron:
    """
    Perceptron class
    """
    def __init__(self, bias=None, weights=None, learning_rate=0.1, n_inputs=2, activation=step):
        """
        class constructor
        :param bias: bias of the perceptron
        :param weights: array of weights for the perceptron
        :param learning_rate: learning rate
        :param n_inputs: number of inputs
        :param activation: activation function for the perceptron
        """
        self.bias = bias if bias is not None else np.random.uniform(-2., 2.)
        self.weights = weights if weights is not None else np.random.uniform(-2., 2., size=n_inputs)
        self.lr = learning_rate
        self.activation = activation

    def feed(self, x):
        """
        given a binary input array, computes the output of the perceptron
        according to its bias and weights
        :param x: binary input [array]
        :return: output of the perceptron
        """
        try:
            assert len(self.weights) == len(x)
            return self.activation(np.array(x), self.weights, self.bias)
        except AssertionError:
            print("feed: mismatched lengths")
            return -1.

    def learn(self, point, desired):
        """
        trains the perceptron according to the point and its desired classification
        :param point: point to train the perceptron
        :param desired: desired output for point
        :return: 1- difference with desired output.
                 2- output obtained by feeding the perceptron
        """
        try:
            assert len(point) == len(self.weights)
            diff = desired - self.feed(point)
            self.weights += self.lr * point * diff
            self.bias += self.lr * diff
            return diff, desired - diff
        except AssertionError:
            print("learn: mismatched lengths")

    def learn_all(self, inputs, outputs):
        """
        given an input, trains the perpectron to learn to classify new points according to
        the desired classifications
        :param inputs: array of points to train the perceptron
        :param outputs: array of desired classifications to train the perceptron
        :return: 1- array of differences between desired and real outputs
                 2- array of classifications given by the perceptron
        """
        precision = np.array([])
        classification = np.array([])
        for ins, outs in zip(inputs, outputs):
            pres, clas = self.learn(ins, outs)
            precision = np.append(precision, pres)
            classification = np.append(classification, clas)
        return precision, classification


class AndPerceptron(Perceptron):
    """
    AndPerceptron class
    class for perceptrons that simulates the AND gate
    """
    def __init__(self):
        super().__init__(-3., [2., 2.])


class OrPerceptron(Perceptron):
    """
    OrPerceptron class
    class for perceptrons that simulates the OR gate
    """
    def __init__(self):
        super().__init__(-1., [2., 2.])


class NandPerceptron(Perceptron):
    """
    NandPerceptron class
    class for perceptrons that simulates the NAND gate
    """
    def __init__(self):
        super().__init__(3., [-2., -2.])


class NotPerceptron(Perceptron):
    """
    NotPerceptron class
    class for perceptrons that simulates the NOT gate
    """
    def __init__(self):
        super().__init__(0.5, [-1.])
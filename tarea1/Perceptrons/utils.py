import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


def sigmoid(inputs, weights, bias):
    """
    applies the sigmoid function to the parameters of the perceptron
    :param inputs: inputs of the perceptron
    :param weights: weights of the perceptron
    :param bias: bias of the perceptron
    :return: output of the perceptron
    """
    z = np.sum(weights * inputs, dtype=np.float128) + bias
    e = np.exp(-z)
    r = 1 / (1 + e)
    return 0. if r < .5 else 1.


def step(inputs, weights, bias):
    """
    applies the step function to the parameters of the perceptron
    :param inputs: inputs of the perceptron
    :param weights: weights of the perceptron
    :param bias: bias of the perceptron
    :return: output of the perceptron
    """
    return 1. if sum(weights * inputs) + bias > 0 else 0.

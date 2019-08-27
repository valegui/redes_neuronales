import numpy as np
from utils import *
from Perceptron import Perceptron
from NeuronLayer import NeuronLayer
from Activation import Step, Sigmoid, Tanh


class NeuralNetwork:
    """
    NeuralNetwork class
    """
    def __init__(self, f, n_n_porcapa, n_capas, o):
        """
        class constructor
        :param f: number of inputs
        :param n_n_porcapa: number of neurons per hidden layer
        :param n_capas: number of hidden layers
        :param o: number of outputs
        """
        layers = [NeuronLayer(n=n_n_porcapa[0], ni=f)]

        for n, ni in zip(n_n_porcapa[1:], n_n_porcapa):
            layers.append(NeuronLayer(n=n, ni=ni))
        layers.append(NeuronLayer(n=o, ni=n_n_porcapa[-1]))

        self.f = f
        self.n_layers = n_capas + 1
        self.layers = np.asarray(layers)

        self.set_last_layer()
        self.set_sibling_layers()

    def set_sibling_layers(self):
        """
        sets next end previous layer for every layer
        :return:
        """
        for i in range(self.n_layers - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])
            self.layers[i + 1].set_prev_layer(self.layers[i])

    def set_last_layer(self):
        """
        sets is_output value as True in the output layer
        :return:
        """
        self.layers[-1].is_output = True

    def feed(self, inputs):
        """
        feeds the neural network, starting from the first hidden
        layer
        :param inputs: input of the network
        :return: the output value
        """
        assert len(inputs) == self.f,\
            f'Mismatched lengths: expected {self.f}, got {len(inputs)}'
        return self.layers[0].feed(np.asarray(inputs))

    def train(self, inputs, expected):
        """
        trains the neural network, with the feed forward and
        the backpropagation stage
        :param inputs: input of the neural network
        :param expected: expected output of the network
        :return: square error after one forward and backward
        """
        _ = self.feed(inputs)
        self.backpropagate_error(np.asarray(expected))
        self.update_weights(inputs)

        err = self.feed(inputs) - np.asarray(expected)
        return np.sum(err**2)

    def backpropagate_error(self, expected):
        """
        backpropagates the error, starting in the output layer
        :param expected: expected output of the network
        :return:
        """
        self.layers[-1].backpropagate_error(expected)

    def update_weights(self, inputs):
        """
        updates the weights, starting in the first layer, for the
        backpropagation stage
        :param inputs: input of the neural network
        :return:
        """
        self.layers[0].update_weights(inputs)

    def load_weights(self, weights):
        """
        load pre computed weights for every layer of the network
        :param weights: new weights
        :return:
        """
        assert len(weights) == self.n_layers,\
            "Length of weights does not match the number of layers"

        for layer, w in zip(self.layers, weights):
            layer.load_weights(w)

    def set_activation(self, activations):
        """
        sets the corresponding activation function to every layer
        of the network
        :param activations: list of activation functions
        :return:
        """
        assert len(activations) == self.n_layers,\
            "Length of activations does not match the number of layers"
        for layer, act in zip(self.layers, activations):
            layer.set_activation(act)

    def set_learning_rate(self, learning_rates):
        """
        sets the corresponding learning rate to every layer of the
        network
        :param learning_rates: list of learning rates
        :return:
        """
        assert len(learning_rates) == self.n_layers,\
            "Length of learning rates does not match the number of layers"
        assert np.issubdtype(np.asarray(learning_rates).dtype, np.number), \
            "Input type not numeric"
        for layer, lr in zip(self.layers, learning_rates):
            layer.set_learning_rate(lr)

    def get_last_activation(self):
        """
        gets the activation object of the output layer
        :return: the activation object
        """
        return self.layers[-1].get_activation()
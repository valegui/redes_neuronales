import numpy as np
from utils import step, sigmoid, tanh
from Activation import Step, Sigmoid, Tanh


class Perceptron:
    """
    Perceptron class
    """
    def __init__(self, bias=None, weights=None, learning_rate=0.1, n_inputs=2, activation=Step()):
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
        self.output = 0.
        self.delta = 1.
        self.lr = learning_rate
        self.activation = activation

    def feed(self, input):
        """
        given an array as an input of the perceptron, returns the
        output of the activation function
        :param input: numeric array, input of the perceptron
        :return: output of the perceptron
        """
        try:
            assert len(self.weights) == len(input),\
                f'Mismatched lengths: expected {len(self.weights)}, got {len(input)}'
            #self.output = self.activation(np.array(input), self.weights, self.bias)
            pre_out = self.activation.apply(np.dot(self.weights, np.array(input)) + self.bias)
            #self.output = 0. if pre_out<0.5 else 1.
            self.output = pre_out
            return self.output
        except AssertionError:
            print("feed: mismatched lengths")
            return

    def train(self, point, desired):
        """
        trains the perceptron according to the point and its desired classification
        :param point: point to train the perceptron
        :param desired: desired output for point
        :return: 1- difference with desired output.
                 2- output obtained by feeding the perceptron
        """
        try:
            assert len(point) == len(self.weights)
            self.delta = desired - self.feed(point)
            self.adjust_weight(point)
            self.adjust_bias()
            return self.delta, desired - self.delta
        except AssertionError:
            print("learn: mismatched lengths")

    def train_all(self, inputs, outputs):
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

    def adjust_bias(self):
        """
        adjutst the bias with the values of delta and the
        learning rate
        :return:
        """
        self.bias += self.lr * self.delta

    def adjust_delta(self, err):
        """
        adjust the value of delta with an error and the derivative
        of the activation function
        :param err: error value
        :return:
        """
        self.delta = err * self.transfer_derivative(self.output)

    def adjust_weight(self, inputs):
        """
        adjust the values of the weights with the values of
        the input data, learning rate and delta
        :param inputs: input data
        :return:
        """
        self.weights += self.lr * inputs * self.delta

    def transfer_derivative(self, output):
        """
        given an output, returns the derivate of the activation
        function with the input that returned that output
        :param output: output of a perceptron
        :return:
        """
        return self.activation.derivate_output(output)

    def set_bias(self, b):
        """
        bias setter
        :param b: new bias
        :return:
        """
        self.bias = b

    def set_delta(self, d):
        """
        delta setter
        :param d: new delta
        :return:
        """
        self.delta = d

    def set_learning_rate(self, lr):
        """
        learning rate setter
        :param lr: new learning rate
        :return:
        """
        self.lr = lr

    def set_activation(self, activation):
        """
        activation function setter
        :param activation: new activation
        :return:
        """
        self.activation = activation

    def set_weights(self, weights):
        """
        weights setter
        :param weights: array with new weights
        :return:
        """
        assert len(weights) == len(self.weights),\
            "Lengths of weights does not match the number of weights per neuron"
        assert np.issubdtype(np.asarray(weights).dtype, np.number), \
            "Input type not numeric"
        self.weights = np.array(weights)

    def get_output(self):
        """
        output getter
        :return: output
        """
        return self.output

    def get_delta(self):
        """
        delta getter
        :return: delta
        """
        return self.delta

    def get_weights(self):
        """
        weights getter
        :return: weights
        """
        return self.weights


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
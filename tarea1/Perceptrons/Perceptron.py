import numpy as np


class Perceptron:
    """
    Perceptron class
    """
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = np.array(weights)

    def feed(self, x):
        """
        given a binary input array, computes the output of the perceptron
        according to its bias and weights
        :param x: binary input [array]
        :return: binary output, -1 if error
        """
        try:
            assert len(self.weights) == len(x)
            return 1. if sum(self.weights * np.array(x)) + self.bias > 0 else 0.
        except AssertionError:
            return -1.


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

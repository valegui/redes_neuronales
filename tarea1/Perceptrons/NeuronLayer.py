from utils import *
from Perceptron import Perceptron
from Activation import Step, Sigmoid, Tanh


class NeuronLayer:
    """
    NeuronLayer class
    """
    def __init__(self, n, ni=0, is_output=False,
                 next_layer=None, prev_layer=None):
        """
        class constructor
        :param n: number of neurons in this layer
        :param ni: number of neurons in the previous layer
        :param is_output: boolean, True if its the output layer
        :param next_layer: next layer in the neural network
        :param prev_layer: previous layer in the neural network
        """
        self.n = n
        self.is_output = is_output
        # neurons
        self.neurons = np.array([Perceptron(activation=Sigmoid(),
                                            n_inputs=ni,
                                            learning_rate=0.5) for i in range(n)])
        # layers
        self.next_layer = next_layer
        self.prev_layer = prev_layer

    def set_next_layer(self, layer):
        """
        next layer setter
        :param layer: new next layer
        :return:
        """
        self.next_layer = layer

    def set_prev_layer(self, layer):
        """
        previous layer setter
        :param layer: new previous layer
        :return:
        """
        self.prev_layer = layer

    def get_outputs(self):
        """
        gets the outputs of every neuron of the layer
        :return: array with the outputs
        """
        output = []
        for neuron in self.neurons:
            output.append(neuron.get_output())
        return np.array(output)

    def get_deltas(self):
        """
        gets the deltas of every neuron of the layer
        :return: array with the deltas
        """
        deltas = []
        for neuron in self.neurons:
            deltas.append(neuron.get_delta())
        return np.array(deltas)

    def get_weights(self):
        """
        gets the weights of every neuron of the layer
        :return: matrix (2d array) with the weights
        """
        w = []
        for neuron in self.neurons:
            w.append(neuron.get_weights())
        return np.vstack(w)

    def feed(self, inputs):
        """
        feeds every perceptron of the layer with inputs and returns
        the output of this layer if its the output layer, otherwise feeds
        the next layer with the output
        :param inputs: numeric array, input of every neuron
        :return: the output of the neural network
        """
        assert np.issubdtype(np.asarray(inputs).dtype, np.number),\
            "Input type not numeric"

        partial_output = np.array([])
        for i in range(self.n):
            partial_output = np.append(partial_output,
                                       self.neurons[i].feed(inputs))

        if self.is_output:
            return partial_output
        else:
            return self.next_layer.feed(partial_output)

    def backpropagate_error(self, expected=None):
        """
        back propagates the error of the neural network
        :param expected: expected output of the neural network
        :return:
        """
        if expected is not None:
            assert np.issubdtype(np.asarray(expected).dtype, np.number), \
                "Input type not numeric"

        if self.is_output:
            self.backpropagate_error_ol(expected)
        else:
            self.backpropagate_error_hl()

    def backpropagate_error_ol(self, expected):
        """
        back propagates the error of the neural network
        for the output layer
        :param expected: expected output of the neural network
        :return:
        """
        exp = np.asarray(expected)
        err = np.subtract(exp, self.get_outputs())

        for i in range(self.n):
            self.neurons[i].adjust_delta(err[i])

        if self.prev_layer is not None:
            self.prev_layer.backpropagate_error()

    def backpropagate_error_hl(self):
        """
        back propagates the error of the neural network
        for the hidden layers
        :return:
        """
        weights = self.next_layer.get_weights()
        deltas = self.next_layer.get_deltas()
        err = np.matmul(deltas, weights)
        for i in range(self.n):
            self.neurons[i].adjust_delta(err[i])

        if self.prev_layer is not None:
            self.prev_layer.backpropagate_error()

    def update_weights(self, init_inputs):
        """
        updates the weights of the neurons of the layer
        for the backpropagation stage. If its not the output layer,
        calls the next layer to update its weights
        :param init_inputs: initial input of the neural network
        :return:
        """
        if self.prev_layer is not None:
            inputs = self.prev_layer.get_outputs()
        else:
            inputs = init_inputs

        for n in self.neurons:
            n.adjust_weight(inputs)
            n.adjust_bias()

        if self.next_layer is not None:
            self.next_layer.update_weights(init_inputs)

    def set_learning_rate(self, learning_rate):
        """
        sets a new learning rate for every neuron of the layer
        :param learning_rate: new learning rate
        :return:
        """
        for neuron in self.neurons:
            neuron.set_learning_rate(learning_rate)

    def set_activation(self, activation):
        """
        sets a new activation function for every neuron of the
        layer
        :param activation: new activation function
        :return:
        """
        for neuron in self.neurons:
            neuron.set_activation(activation)

    def load_weights(self, weights):
        """
        load pre computed weights for every neuron of the layer
        :param weights: new weights
        :return:
        """
        assert len(weights) == self.n ,\
            "Lengths of weights does not match the number of neurons on the layer"
        for neuron, weight in zip(self.neurons, weights):
            neuron.set_weights(weight)

    def get_activation(self):
        """
        gets the activation function of the layer
        :return: the activation function class
        """
        return self.neurons[0].activation

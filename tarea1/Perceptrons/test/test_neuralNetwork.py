from unittest import TestCase
from NeuralNetwork import NeuralNetwork
from Activation import Sigmoid, Tanh, Step
from NeuronLayer import NeuronLayer
import numpy as np


class TestNeuralNetwork(TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(3, [4, 5], 2, 3)

    def test_init_sibling_layers(self):
        self.assertEqual(self.nn.layers[0], self.nn.layers[1].prev_layer)
        self.assertEqual(self.nn.layers[1], self.nn.layers[2].prev_layer)
        self.assertEqual(self.nn.layers[1], self.nn.layers[0].next_layer)
        self.assertEqual(self.nn.layers[2], self.nn.layers[1].next_layer)
        self.assertIsNone(self.nn.layers[0].prev_layer)
        self.assertIsNone(self.nn.layers[2].next_layer)

    def test_init_last_layer(self):
        self.assertFalse(self.nn.layers[0].is_output)
        self.assertFalse(self.nn.layers[1].is_output)
        self.assertTrue(self.nn.layers[2].is_output)

    def test_feed(self):
        f = self.nn.feed([0.3, 0.5, 0.4])
        self.assertTrue(np.issubdtype(f.dtype, np.number))
        self.assertEqual(3, len(f))

    def test_set_activation(self):
        self.assertEqual(Sigmoid().__class__, self.nn.layers[0].get_activation().__class__)
        self.assertEqual(Sigmoid().__class__, self.nn.layers[1].get_activation().__class__)
        self.assertEqual(Sigmoid().__class__, self.nn.layers[2].get_activation().__class__)
        self.nn.set_activation([Tanh(), Step(), Tanh()])
        self.assertEqual(Tanh().__class__, self.nn.layers[0].get_activation().__class__)
        self.assertEqual(Step().__class__, self.nn.layers[1].get_activation().__class__)
        self.assertEqual(Tanh().__class__, self.nn.layers[2].get_activation().__class__)

    def test_set_learning_rate(self):
        self.assertEqual(0.5, self.nn.layers[0].neurons[0].lr)
        self.assertEqual(0.5, self.nn.layers[0].neurons[1].lr)
        self.assertEqual(0.5, self.nn.layers[0].neurons[2].lr)
        self.assertEqual(0.5, self.nn.layers[0].neurons[3].lr)
        self.assertEqual(0.5, self.nn.layers[1].neurons[0].lr)
        self.assertEqual(0.5, self.nn.layers[1].neurons[1].lr)
        self.assertEqual(0.5, self.nn.layers[1].neurons[2].lr)
        self.assertEqual(0.5, self.nn.layers[1].neurons[3].lr)
        self.assertEqual(0.5, self.nn.layers[1].neurons[4].lr)
        self.assertEqual(0.5, self.nn.layers[2].neurons[0].lr)
        self.assertEqual(0.5, self.nn.layers[2].neurons[1].lr)
        self.assertEqual(0.5, self.nn.layers[2].neurons[2].lr)
        self.nn.set_learning_rate([0.15, 0.25, 0.35])
        self.assertEqual(0.15, self.nn.layers[0].neurons[0].lr)
        self.assertEqual(0.15, self.nn.layers[0].neurons[1].lr)
        self.assertEqual(0.15, self.nn.layers[0].neurons[2].lr)
        self.assertEqual(0.15, self.nn.layers[0].neurons[3].lr)
        self.assertEqual(0.25, self.nn.layers[1].neurons[0].lr)
        self.assertEqual(0.25, self.nn.layers[1].neurons[1].lr)
        self.assertEqual(0.25, self.nn.layers[1].neurons[2].lr)
        self.assertEqual(0.25, self.nn.layers[1].neurons[3].lr)
        self.assertEqual(0.25, self.nn.layers[1].neurons[4].lr)
        self.assertEqual(0.35, self.nn.layers[2].neurons[0].lr)
        self.assertEqual(0.35, self.nn.layers[2].neurons[1].lr)
        self.assertEqual(0.35, self.nn.layers[2].neurons[2].lr)

    def test_get_last_activation(self):
        self.assertEqual(Sigmoid().__class__, self.nn.get_last_activation().__class__)

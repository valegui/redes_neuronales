from unittest import TestCase
from NeuronLayer import NeuronLayer
from Activation import Sigmoid, Step, Tanh
import numpy as np


class TestNeuronLayer(TestCase):
    def setUp(self):
        self.layer1 = NeuronLayer(n=4, ni=5)
        self.layer2 = NeuronLayer(n=6, ni=4)

    def test_set_next_layer(self):
        self.assertIsNone(self.layer1.next_layer)
        self.layer1.set_next_layer(self.layer2)
        self.assertEqual(self.layer2, self.layer1.next_layer)

    def test_set_prev_layer(self):
        layer1 = NeuronLayer(n=6, ni=4)
        layer2 = NeuronLayer(n=4, ni=5)
        self.assertIsNone(self.layer2.prev_layer)
        self.layer2.set_prev_layer(self.layer1)
        self.assertEqual(self.layer1, self.layer2.prev_layer)

    def test_get_outputs(self):
        o = self.layer1.get_outputs()
        self.assertEqual(4, len(o))
        self.assertTrue(np.issubdtype(o.dtype, np.number))

    def test_get_deltas(self):
        d = self.layer1.get_deltas()
        self.assertEqual(4, len(d))
        self.assertTrue(np.issubdtype(d.dtype, np.number))

    def test_get_weights(self):
        w = self.layer1.get_weights()
        self.assertTrue(np.issubdtype(w.dtype, np.number))
        self.assertEqual(4, len(w))
        self.assertEqual(5, len(w[0]))
        self.assertEqual(5, len(w[1]))
        self.assertEqual(5, len(w[2]))
        self.assertEqual(5, len(w[3]))

    def test_set_learning_rate(self):
        self.assertEqual(0.5, self.layer1.neurons[0].lr)
        self.assertEqual(0.5, self.layer1.neurons[1].lr)
        self.assertEqual(0.5, self.layer1.neurons[2].lr)
        self.assertEqual(0.5, self.layer1.neurons[3].lr)
        self.layer1.set_learning_rate(0.2)
        self.assertEqual(0.2, self.layer1.neurons[0].lr)
        self.assertEqual(0.2, self.layer1.neurons[1].lr)
        self.assertEqual(0.2, self.layer1.neurons[2].lr)
        self.assertEqual(0.2, self.layer1.neurons[3].lr)

    def test_set_activation(self):
        self.assertEqual(Sigmoid().__class__, self.layer1.get_activation().__class__)
        self.layer1.set_activation(Tanh())
        self.assertEqual(Tanh().__class__, self.layer1.get_activation().__class__)

    def test_load_weights(self):
        wi = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
              [10, 20, 30, 40, 50], [50, 60, 70, 80, 90]]
        self.layer1.load_weights(wi)
        w = self.layer1.get_weights()
        self.assertTrue(np.issubdtype(w.dtype, np.number))
        self.assertEqual(4, len(w))
        self.assertTrue(np.allclose([1, 2, 3, 4, 5], w[0]))
        self.assertTrue(np.allclose([6, 7, 8, 9, 10], w[1]))
        self.assertTrue(np.allclose([10, 20, 30, 40, 50], w[2]))
        self.assertTrue(np.allclose([50, 60, 70, 80, 90], w[3]))

    def test_get_activation(self):
        layer = NeuronLayer(n=4, ni=5)
        self.assertEqual(Sigmoid().__class__, self.layer1.get_activation().__class__)

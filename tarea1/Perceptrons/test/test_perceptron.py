from unittest import TestCase
import numpy as np

from Perceptron import Perceptron
from Perceptron import AndPerceptron
from Perceptron import NandPerceptron
from Perceptron import OrPerceptron
from Perceptron import NotPerceptron


class TestPerceptron(TestCase):
    def test_feed(self):
        p1 = Perceptron(0., [1.])
        self.assertEqual(0., p1.feed([0.]))
        self.assertEqual(1., p1.feed([1.]))
        p2 = Perceptron(-5., [3., 2., 4.])
        self.assertEqual(0., p2.feed([0., 0., 1.]))
        self.assertEqual(0., p2.feed([0., 1., 0.]))
        self.assertEqual(1., p2.feed([0., 1., 1.]))
        self.assertEqual(0., p2.feed([1., 0., 0.]))
        self.assertEqual(1., p2.feed([1., 0., 1.]))
        self.assertEqual(0., p2.feed([1., 1., 0.]))
        self.assertEqual(1., p2.feed([1., 1., 1.]))

    def test_feed_error(self):
        p = Perceptron(0., [1., 1.])
        self.assertEqual(-1., p.feed([0.]))

    def test_learn(self):
        p = Perceptron(1., [1., 1.], .1)
        # desired = 0, real = 0, diff = 0
        point1 = np.array([-1., -1.])
        desired1 = 0.
        diff1, real1 = p.learn(point1, desired1)
        self.assertEqual(0., diff1)
        self.assertEqual(0., real1)
        self.assertEqual(1., p.weights[0])
        self.assertEqual(1., p.weights[1])
        self.assertEqual(1., p.bias)
        # desired = 1, real = 1, diff = 0
        point2 = np.array([-1., 1.])
        desired2 = 1.
        diff2, real2 = p.learn(point2, desired2)
        self.assertEqual(0., diff2)
        self.assertEqual(1., real2)
        self.assertEqual(1., p.weights[0])
        self.assertEqual(1., p.weights[1])
        self.assertEqual(1., p.bias)
        # desired = 0, real = 1, diff = -1
        point3 = np.array([-10, 10])
        desired3 = 0.
        diff3, real3 = p.learn(point3, desired3)
        self.assertEqual(-1., diff3)
        self.assertEqual(1., real3)
        self.assertEqual(2., p.weights[0])
        self.assertEqual(0., p.weights[1])
        self.assertEqual(.9, p.bias)
        # desired = 1, real = 0, diff = 1
        point4 = np.array([-1, 10])
        desired4 = 1.
        diff4, real4 = p.learn(point4, desired4)
        self.assertEqual(1., diff4)
        self.assertEqual(0., real4)
        self.assertEqual(1.9, p.weights[0])
        self.assertEqual(1., p.weights[1])
        self.assertEqual(1., p.bias)

    def test_learn_error(self):
        p = Perceptron(1., [1., 1.], .1)
        point = np.array([1., 2., 3.])
        desired = 0.
        self.assertIsNone(p.learn(point, desired))

    def test_learn_all(self):
        inputs = np.array([[-1., -1.],
                  [-1., 1.],
                  [-10, 10],
                  [-1., 1]])
        outputs = np.array([0., 1., 0., 1.])
        p = Perceptron(1., [1., 1.], .1)
        res1, res2 = p.learn_all(inputs, outputs)
        self.assertEqual(4, len(res1))
        self.assertEqual(4, len(res2))
        # differences
        self.assertEqual(0., res1[0])
        self.assertEqual(0., res1[1])
        self.assertEqual(-1., res1[2])
        self.assertEqual(1., res1[3])
        # classifications done by the perceptron
        self.assertEqual(0., res2[0])
        self.assertEqual(1., res2[1])
        self.assertEqual(1., res2[2])
        self.assertEqual(0., res2[3])
        # final values of weights and bias
        self.assertEqual(1.9, p.weights[0])
        self.assertEqual(.1, p.weights[1])
        self.assertEqual(1., p.bias)


class TestAndPerceptron(TestCase):
    def test_feed(self):
        p = AndPerceptron()
        self.assertEqual(0., p.feed([0., 0.]))
        self.assertEqual(0., p.feed([0., 1.]))
        self.assertEqual(0., p.feed([1., 0.]))
        self.assertEqual(1., p.feed([1., 1.]))


class TestOrPerceptron(TestCase):
    def test_feed(self):
        p = OrPerceptron()
        self.assertEqual(0., p.feed([0., 0.]))
        self.assertEqual(1., p.feed([0., 1.]))
        self.assertEqual(1., p.feed([1., 0.]))
        self.assertEqual(1., p.feed([1., 1.]))


class TestNandPerceptron(TestCase):
    def test_feed(self):
        p = NandPerceptron()
        self.assertEqual(1., p.feed([0., 0.]))
        self.assertEqual(1., p.feed([0., 1.]))
        self.assertEqual(1., p.feed([1., 0.]))
        self.assertEqual(0., p.feed([1., 1.]))


class TestNotPerceptron(TestCase):
    def test_feed(self):
        p = NotPerceptron()
        self.assertEqual(1., p.feed([0.]))
        self.assertEqual(0., p.feed([1.]))

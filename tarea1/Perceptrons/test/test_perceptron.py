from unittest import TestCase

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

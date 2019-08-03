from unittest import TestCase
import numpy as np

from SummingGate import SummingGate


class TestSummingGate(TestCase):
    def test_sum_two(self):
        gate = SummingGate()
        self.assertEqual(2, len(gate.sum_two(0., 0.)))
        self.assertEqual(0., gate.sum_two(0., 0.)[0])
        self.assertEqual(0., gate.sum_two(0., 0.)[1])
        self.assertEqual(1., gate.sum_two(1., 0.)[0])
        self.assertEqual(0., gate.sum_two(1., 0.)[1])
        self.assertEqual(1., gate.sum_two(0., 1.)[0])
        self.assertEqual(0., gate.sum_two(0., 1.)[1])
        self.assertEqual(0., gate.sum_two(1., 1.)[0])
        self.assertEqual(1., gate.sum_two(1., 1.)[1])


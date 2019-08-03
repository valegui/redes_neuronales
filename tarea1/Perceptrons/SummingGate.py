import numpy as np
from Perceptron import NandPerceptron


class SummingGate:
    """
    SummingGate class
    class that simulates a circuit which adds two bits
    """
    def __init__(self):
        self.nand = NandPerceptron()

    def sum_two(self, x1, x2):
        """
        adds two bits
        :param x1: first bit to add
        :param x2: second bit to add
        :return: numpy array with sum bit and carry bit
        """
        direct_nand = self.nand.feed([x1, x2])
        direct_nand_x1 = self.nand.feed([x1, direct_nand])
        direct_nand_x2 = self.nand.feed([x2, direct_nand])
        sum_bit = self.nand.feed([direct_nand_x1, direct_nand_x2])
        carry_bit = self.nand.feed([direct_nand, direct_nand])
        return np.array([sum_bit, carry_bit])

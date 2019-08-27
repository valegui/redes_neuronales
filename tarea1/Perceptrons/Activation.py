import numpy as np 


class Step:
    def apply(self, x):
        return 0 if x < 0 else 1
    
    def derivative(self, x):
        return 0

    def derivate_output(self, x):
        return 0

    def to_bin(self, x):
        return np.vectorize(lambda y:y)(x)


class Sigmoid :
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.apply(x) * (1 - self.apply(x))

    def derivate_output(self, x):
        return x * (1. - x)

    def to_bin(self, x):
        return np.vectorize(lambda y: 0. if y < 0.5 else 1.)(x)


class Tanh:
    def apply(self, x):
        pe = np.exp(x)
        ne = np.exp(-x)
        return (pe - ne) / (pe + ne)

    def derivative(self, x):
        return 1 - self.apply(x)**2

    def derivate_output(self, x):
        return 1. - x**2

    def to_bin(self, x):
        return np.vectorize(lambda y: 0. if y <=0. else 1.)(x)
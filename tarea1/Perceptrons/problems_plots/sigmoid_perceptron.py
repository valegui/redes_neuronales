from Perceptron import Perceptron
from utils import *


def and_data():
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    classification = np.array([0, 0, 0, 1])
    return points, classification


def or_data():
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    classification = np.array([0, 1, 1, 1])
    return points, classification


def nand_data():
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    classification = np.array([1, 1, 1, 0])
    return points, classification


def plot_precision(precision, learning_rate=0.1):
    """
    plots the precision vs number of training curve to a given
    learning rate
    :param precision: array of precision values for every training
    :param learning_rate: learning rate of the perceptron
    :return:
    """
    plt.plot(precision, '-b')
    plt.ylim(0, 1)
    plt.title(f'Number of training vs precision\n Learning rate = {learning_rate}')
    plt.show()


if __name__ == "__main__":
    q = 1000
    trainings = 10
    op = 'AND'
    op_data = {'AND': and_data,
               'OR': or_data,
               'NAND': nand_data}
    for key in op_data.keys():
        for lr in np.arange(0.1, 0.92, 0.1):
            p = Perceptron(learning_rate=lr, activation=sigmoid)
            points, classification = op_data[key]()
            precision_training = np.array([])
            # train the perceptron
            for i in range(trainings):
                local_precision, perceptron_out = p.learn_all(points, classification)
                precision_training = np.append(precision_training, (q - np.count_nonzero(local_precision != 0)) / q)
            plot_precision(precision_training, learning_rate=lr)
            #print(f'{key} para 1 y 0  = {p.feed([1, 0])}')
            #print(f'{key} para 0 y 0  = {p.feed([0, 0])}')
            #print(f'{key} para 1 y 1  = {p.feed([1, 1])}')
            #print(f'{key} para 0 y 1  = {p.feed([0, 1])}')

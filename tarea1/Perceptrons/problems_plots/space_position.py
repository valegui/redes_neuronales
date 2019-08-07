import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron


def generate_points_2d(xlim=50, ylim=60, qty=100):
    """
    generates 2 arrays of points
    :param xlim: limit for the values in the x axis
    :param ylim: limit for the values in the y axis
    :param qty: quantity of points to generate
    :return: [1] array of numbers between -xlim and xlim
             [2] array of numbers between -ylim and ylim
    """
    x = np.random.uniform(-xlim, xlim, size=qty)
    y = np.random.uniform(-ylim, ylim, size=qty)
    return x, y


def generate_curve_2d(low=-10, high=10):
    """
    generates the values for the slope and the intercept
    of the line equation
    ax + b = y
    :param low: lower limit for the number generated
    :param high: upper limit for the number generated
    :return: array of values of the line equation
    """
    return np.random.uniform(low, high, size=2)


def classify_point_2d(x, y, a, b):
    """
    classifies the point (x, y) with 1 if its above the line
    or 0 if its below
    :param x: first coordinate
    :param y: second coordinate
    :param a: slope
    :param b: intercept
    :return: classification
    """
    return 1. if x * a - y > b else 0.


def generate_points_classification_2d(curve, xlim=50, ylim=60, qty=100):
    """

    :param curve: array with slope-intercept values of the line equation
    :param xlim: limit for the values in the x axis
    :param ylim: limit for the values in the y axis
    :param qty: quantity of points to generate
    :return: [1] array of points (x, y)
             [2] array of classifications for every point of [1]
    """
    X, Y = generate_points_2d(xlim, ylim, qty)
    classif = []
    XY = []
    for x, y in zip(X, Y):
        XY.append(np.array([x, y]))
        classif.append(classify_point_2d(x, y, curve[0], curve[1]))
    return np.array(XY), np.array(classif)


def plot_curve_classification(curve, points, classification, xlim=51, ylim=80, n_training=0, learning_rate=0.1):
    """
    plots the line that divides the plane and the points generated with
    its (obtained via perceptron) classifications
    :param curve: array with slope-intercept values of the line equation
    :param points: array of (x, y) points to plot
    :param classification: array of classifications to plot
    :param xlim: limit for the values in the x axis
    :param ylim: limit for the values in the y axis
    :param n_training: number of trainings
    :param learning_rate: learning rate of the perceptron
    :return:
    """
    x_axis = np.arange(-xlim, xlim, 0.01)
    y_axis = x_axis * curve[0] + curve[1]
    plt.ylim([-ylim, ylim])
    plt.plot(x_axis, y_axis, 'g')
    for point, classif in zip(points, classification):
        if classif > 0.5:
            plt.plot(point[0], point[1], '*b')
        else:
            plt.plot(point[0], point[1], '*r')
    plt.title(f'Point classification with {n_training} trainings\n Learning rate = {learning_rate}')
    plt.show()


def plot_precision(precision, learning_rate=0.1):
    """
    plots the precision vs number of training curve to a given
    learning rate
    :param precision: array of precision values for every training
    :param learning_rate: learning rate of the perceptron
    :return:
    """
    plt.plot(precision, '-b')
    plt.title(f'Number of training vs precision\n Learning rate = {learning_rate}')
    plt.show()


if __name__ == "__main__":
    # generate variables for every learning rate
    curve = generate_curve_2d()
    q = 1100
    points, classification = generate_points_classification_2d(curve, qty=q)
    trainings = 52
    plt_classification = False  # True to generate classification plots

    # different learning rates
    for lr in np.arange(0.1, 0.92, 0.1):
        p = Perceptron(learning_rate=lr)
        precision_training = np.array([])
        # train the perceptron
        for i in range(trainings):
            local_precision, perceptron_out = p.learn_all(points, classification)
            precision_training = np.append(precision_training, (q - np.count_nonzero(local_precision != 0)) / q)
            if plt_classification and (i == 0 or i == 20 or i == 50):
                # plot the perceptron output (classification)
                plot_curve_classification(curve, points, perceptron_out, n_training=i)
            # plot precision
        plot_precision(precision_training, learning_rate=lr)


from utils import *
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from Activation import Step, Sigmoid, Tanh


def train_epochs(NN, xTrain, xTest, yTrain, yTest, n_epochs):
    """
    Trains the neural network for the indicated number of epochs
    :param NN: Neural Network
    :param xTrain: training attributes
    :param xTest: testing attributes
    :param yTrain: training classifications
    :param yTest: testing classifications
    :param n_epochs: number of epochs
    :return: 1- errors per epoch array
             2- accuracy per epoch array
    """
    l = len(xTrain)
    l_test = len(xTest)

    err = []
    ac = []
    print(f'Entrenando red para {n_epochs} epochs')
    for i in range(n_epochs):
        print(f'-------- epoch {i}')
        err.append(epoch(NN, l, xTrain, yTrain))
        ac.append(hits(NN, xTest, yTest, l_test)[1])
    return err, ac


def plot_err_acc(error, accuracy, epochs):
    """
    Plots error and accuracy for a number of epochs
    :param error: error array
    :param accuracy: accuracy array
    :param epochs: number of epochs
    :return:
    """
    plt.plot(np.arange(start=1,stop=len(error)+1,step=1),error)
    plt.title(f'Porcentaje de error por epoca \n {epochs} epochs')
    plt.ylabel('Porcentaje de error')
    plt.xlabel('Epoca')
    plt.show()

    plt.plot(np.arange(start=1,stop=len(accuracy)+1,step=1), accuracy)
    plt.title(f'Porcentaje de aciertos por epoca. \n {epochs} epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.xlabel('Epoca')
    plt.show()


if __name__ == "__main__":
    # Obtener data
    dataset = np.loadtxt("./training_data/seeds_dataset.txt")
    dataset_attr = normalize_dataset(get_attributes(dataset))
    dataset_class, hots = hot_encode(get_classification(dataset))

    # Split data
    xTrain, xTest, yTrain, yTest = train_test_split(dataset_attr, dataset_class, test_size=0.26, random_state=0)

    # Crear red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Sigmoid(),Sigmoid(),Sigmoid()])
    nn.set_learning_rate([0.6,0.6,0.6])

    # Entrenar
    epochs = 50
    err, ac = train_epochs(nn, xTrain, xTest, yTrain, yTest, epochs)
    plot_err_acc(err, ac, epochs)

    # Confusion Matrix
    yPred = []
    for x in xTest:
        yPred.append(nn.feed(x))

    cm = confusion_matrix_(nn, hots, yTest, yPred, plot=True)
    print('Confusion Matrix\n', cm)

    # Otra red neuronal

    # Crear red neuronal
    nn = NeuralNetwork(7, [5, 4, 6], 3, 3)
    nn.set_activation([Sigmoid(),Sigmoid(),Tanh(),Sigmoid()])
    nn.set_learning_rate([0.6,0.6,0.5,0.6])

    # Entrenar
    err, ac = train_epochs(nn, xTrain, xTest, yTrain, yTest, epochs)
    plot_err_acc(err, ac, epochs)

    # Confusion Matrix
    yPred = []
    for x in xTest:
        yPred.append(nn.feed(x))

    cm = confusion_matrix_(nn, hots, yTest, yPred, plot=True)
    print('Confusion Matrix\n', cm)

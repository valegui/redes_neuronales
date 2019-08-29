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
    plt.title(f'MSE por epoca \n {epochs} epochs')
    plt.ylabel('Porcentaje de error')
    plt.xlabel('Epoca')
    plt.show()

    plt.plot(np.arange(start=1,stop=len(accuracy)+1,step=1), accuracy)
    plt.title(f'Porcentaje de aciertos por epoca. \n {epochs} epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.xlabel('Epoca')
    plt.show()



def train_plot_cm(NN, xTrain, xTest, yTrain, yTest, epochs, encoding_dict):
    """
    Trains a neural network, plots the error and accuracy for a
    number of epochs and plots the confussion matrix
    :param NN: neural network
    :param xTrain: training attributes
    :param xTest: testing attributes
    :param yTrain: training classifications
    :param yTest: testing classifications
    :param epochs: number of epochs
    :param encoding_dict: dict with the attributes and codes
    :return:
    """
    err, ac = train_epochs(NN, xTrain, xTest, yTrain, yTest, epochs)
    plot_err_acc(err, ac, epochs)

    yPred = []
    for x in xTest:
        yPred.append(NN.feed(x))

    cm = confusion_matrix_(NN, encoding_dict, yTest, yPred, plot=True)
    print('Confusion Matrix\n', cm)


def twohl(xTrain, xTest, yTrain, yTest, hots):
    print('EXPERIMENTO CON DOS CAPAS ESCONDIDAS')
    # Primera red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    # Segunda red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Tanh(), Tanh(), Tanh()])
    nn.set_learning_rate([0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    # Tercera red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.3, 0.3, 0.3])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    # Cuarta red neuronal
    nn = NeuralNetwork(7, [10, 10], 2, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    # Quinta red neuronal
    nn = NeuralNetwork(7, [1, 2], 2, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)


def tanhexp(xTrain, xTest, yTrain, yTest, hots):
    print('EXPERIMENTO CON FUNCION DE ACTIVACION TANH')
    # Primera red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Tanh(), Tanh(), Tanh()])
    nn.set_learning_rate([0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=60, encoding_dict=hots)

    # Segunda red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Tanh(), Tanh(), Tanh()])
    nn.set_learning_rate([0.3, 0.3, 0.3])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=60, encoding_dict=hots)

    # Tercera red neuronal
    nn = NeuralNetwork(7, [5, 4], 2, 3)
    nn.set_activation([Tanh(), Tanh(), Tanh()])
    nn.set_learning_rate([0.2, 0.2, 0.2])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=60, encoding_dict=hots)


def fourhl(xTrain, xTest, yTrain, yTest, hots):
    print('EXPERIMENTO CON CUATRO CAPAS ESCONDIDAS')
    nn = NeuralNetwork(7, [5, 5, 4, 4], 4, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    nn = NeuralNetwork(7, [2, 2, 2, 2], 4, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)

    nn = NeuralNetwork(7, [10, 10, 9, 8], 4, 3)
    nn.set_activation([Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()])
    nn.set_learning_rate([0.6, 0.6, 0.6, 0.6, 0.6])

    train_plot_cm(nn, xTrain, xTest, yTrain, yTest,
                  epochs=50, encoding_dict=hots)


if __name__ == "__main__":
    # Obtener data
    dataset = np.loadtxt("Perceptrons/training_data/seeds_dataset.txt")
    dataset_attr = normalize_dataset(get_attributes(dataset))
    dataset_class, hots = hot_encode(get_classification(dataset))

    # Split data
    xTrain, xTest, yTrain, yTest = train_test_split(dataset_attr, dataset_class, test_size=0.26, random_state=0)

    runoptions = {'1': twohl,
                  '2': tanhexp,
                  '3': fourhl,
                  }
    try:
        print("OPCIONES DE PROGRAMA:")
        print("1 : Redes neuronales con 2 capas escondidas")
        print("2 : Redes neuronales con solo Tanh y diferentes learning rate")
        print("3 : Redes neuronales con 4 capas escondidas")
        print()
        num = input("Ingrese el numero del programa: ")
        print()
        runoptions[num](xTrain, xTest, yTrain, yTest, hots)
    except:
        print("\nFin del programa")

from three_layer_neural_network import NeuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200,noise = 0.1)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Layer(object):
    def __init__(self, input_dim, output_dim,actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))


    def feedforward(self, X, actFun):
        self.X = X
        self.z = X @ self.W + self.b
        self.a = actFun(self.z)
        return self.a

    def backprop(self, ddx_prev,diff_actFun):
        delta = ddx_prev * diff_actFun(self.z)
        dW = np.transpose(self.X) @ delta
        db = np.sum(delta, axis=0)
        ddx = (delta @ np.transpose(self.W))
        return ddx, dW, db


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,n_layers, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, epsilon=0.01,seed=0):
        self.n_layers = n_layers
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        np.random.seed(seed)
        self.layers = []
        self.layers.append(Layer(self.nn_input_dim, self.nn_hidden_dim))
        for i in range(1, n_layers-1):
            self.layers.append(Layer(self.nn_hidden_dim, self.nn_hidden_dim))
        self.layers.append(Layer(self.nn_hidden_dim, self.nn_output_dim))


    def feedforward(self, X, actFun):
        for i in range(self.n_layers):
            X = self.layers[i].feedforward(X, actFun)
        self.probs = np.exp(self.layers[-1].z) / np.sum(np.exp(self.layers[-1].z), axis=1, keepdims=True)
        return None

    def backprop(self, X, y):
        num_examples = len(X)
        delta1 = self.probs
        delta1[range(num_examples), y] -= 1
        temp = self.layers[-1].backprop(delta1, lambda x: 1)
        ddx_prev = temp[0]
        self.layers[-1].dW = temp[1]
        self.layers[-1].db = temp[2]
        for i in reversed(range(0, self.n_layers-1)):
            newtemp = self.layers[i].backprop(ddx_prev, lambda x: self.diff_actFun(x, self.actFun_type))
            ddx_prev = newtemp[0]
            self.layers[i].dW = newtemp[1]
            self.layers[i].db = newtemp[2]


    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss
        data_loss = -np.sum(np.log(self.probs[range(num_examples),y]))
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * sum([np.sum(np.square(self.layers[i].W)) for i in range(self.n_layers)])
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y,  num_passes=20000, print_loss=True):
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            self.backprop(X, y)
            for j in range(self.n_layers):
                self.layers[j].dW += self.reg_lambda * self.layers[j].W
                self.layers[j].W += -self.epsilon * self.layers[j].dW
                self.layers[j].b += -self.epsilon * self.layers[j].db
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))



def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    model = DeepNeuralNetwork(n_layers=3,nn_input_dim=2, nn_hidden_dim=10 , nn_output_dim=2,  actFun_type='tanh')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()
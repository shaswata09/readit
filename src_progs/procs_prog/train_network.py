import numpy as np
import random

from src_progs.procs_prog.read_network import read_network
from src_progs.procs_prog.update_network import update_network
from src_progs.procs_prog.read_train_file import read_train_file
from src_progs.procs_prog.read_test_file import read_test_file


class Network(object):

    def __init__(self, net):
        self.num_layers = len(net.sizes)
        self.sizes = net.sizes
        self.biases = net.biases
        self.weights = net.weights


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                success=self.evaluate(test_data)
                success_rate = float(success*100)/n_test
                print("After Epoch {0:>3}: {1:>5} / {2:<5} Equivalent to {3:<18} (%)Percent.".format(j, success, n_test, success_rate))
            else:
                print("After Epoch {0:>3} complete".format(j))
            if j%10==0 :
                update_network(self, success_rate)

        return self.sizes, self.biases, self.weights, success_rate


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        out_lst = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                   "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                   "U", "V", "W", "X", "Y", "Z",
                   "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u", "v", "w", "x", "y", "z"]
        test_results = [(out_lst[np.argmax(self.feedforward(x))], y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def train_network():
    net = read_network()
    train_img_list = read_train_file()
    test_img_list = read_test_file()

    print("Starting to train neural network...")
    epochs = int(input("Enter number of epochs (e.g: 30)...:"))
    mini_batch_size = int(input("Enter mini_batch_size (e.g: 10)...:"))
    eta=float(1.5)
    network = Network(net)

    # starting to train network using SGD
    print("Starting training using SDG.... It will take some time... Please wait...")

    net.sizes,net.biases,net.weights,success_rate = network.SGD(train_img_list,epochs,mini_batch_size,eta,test_img_list)

    print("Network training successful... Now new optimized network will be written in file.")

    update_network(net,success_rate)

    print(f"Neural is now successfully trained and updated with {success_rate:<18} % success rate.")



if __name__ == "__main__":
    train_network()


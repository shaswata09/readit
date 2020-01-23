import numpy as np
import pickle


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def update_network(net, success_rate):
    net_biases_path = "/home/leonardo/Desktop/readit_test/trained_network/bias.pkl"
    net_weights_path = "/home/leonardo/Desktop/readit_test/trained_network/weights.pkl"
    net_size_path = "/home/leonardo/Desktop/readit_test/trained_network/size.pkl"

    print("Strating to update Neural Network files...")

    with open(net_size_path, 'rb') as f:
        size = pickle.load(f)
    prev_success_rate=size[1]
    size[1]=success_rate
    with open(net_size_path, 'wb') as f:
        pickle.dump(size, f)

    # success rate is updated and now updating other files.

    with open(net_biases_path, 'wb') as f:
        pickle.dump(net.biases, f)
    with open(net_weights_path, 'wb') as f:
        pickle.dump(net.weights, f)

    print(f"Neural Network has been Successfully updated from {prev_success_rate:<18} % to {success_rate:<18} % success rate in its corresponding files.")


if __name__ == "__main__":
    update_network()
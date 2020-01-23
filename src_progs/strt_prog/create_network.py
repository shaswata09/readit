import numpy as np
import pickle


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

def create_network(size):
    net_biases_path = "/home/leonardo/Desktop/readit_test/trained_network/bias.pkl"
    net_weights_path = "/home/leonardo/Desktop/readit_test/trained_network/weights.pkl"
    net_size_path = "/home/leonardo/Desktop/readit_test/trained_network/size.pkl"

    print("Strating Neural Network Creation...")

    net = Network(size[0])
    with open(net_size_path, 'wb') as f:
        pickle.dump(size, f)
    with open(net_biases_path, 'wb') as f:
        pickle.dump(net.biases, f)
    with open(net_weights_path, 'wb') as f:
        pickle.dump(net.weights,f)

    print("Neural Network has been Successfully created and saved to its corresponding files.")

if __name__ == "__main__":
    success_rate=float(0)
    size = [[1600,600,100,62],success_rate]
    create_network(size)
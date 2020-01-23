import pickle


def network_status():
    net_size_path = "/home/leonardo/Desktop/readit_test/trained_network/size.pkl"
    with open(net_size_path, 'rb') as f:
        size = pickle.load(f)
    print(f"The network size is: {size[0]}.")
    print(f"The accuracy of the network is : {size[1]} %.")

if __name__ == "__main__" :
    network_status()
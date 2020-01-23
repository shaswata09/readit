import pickle


def read_test_file():

    prcsd_img_file = "/home/leonardo/Desktop/readit_test/test_dataset/processed_img_file.pkl"

    print("Strating to read testing file...")

    with open(prcsd_img_file, 'rb') as f:
        test_img_list = pickle.load(f)

    print("Testing file has been Successfully read from its corresponding file.")

    return test_img_list


if __name__ == "__main__":
    read_test_file()
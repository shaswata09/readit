import pickle


def read_train_file():

    prcsd_img_file = "/home/leonardo/Desktop/readit_test/training_dataset/processed_img_file.pkl"

    print("Strating to read training file...")

    with open(prcsd_img_file, 'rb') as f:
        train_img_list = pickle.load(f)

    print("Training file has been Successfully read from its corresponding file.")

    return train_img_list


if __name__ == "__main__":
    read_train_file()
import cv2
import numpy as np
import glob
import pickle


def generate_test_file():
    prcsd_img_list = (glob.glob("/home/leonardo/Desktop/readit_test/test_dataset/processed_img/*"))
    prcsd_img_file = "/home/leonardo/Desktop/readit_test/test_dataset/processed_img_file.pkl"
    train_img_list = list()

    for path in prcsd_img_list:
        imgpath = path
        out_val = str(path[62])
        img = cv2.imread(imgpath, 0)
        img = np.reshape(img, (1600, 1))

        img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')


        for i in range(img.shape[0]):
            img_arr[i] = (float(img[i]/256))

        img_res= out_val
        img_tup = (img_arr, img_res)
        train_img_list.append(img_tup)

    with open(prcsd_img_file, 'wb') as f:
        pickle.dump(train_img_list, f)

    print(str(len(train_img_list)) + " images testing list is created.")
    print("The list contain tuples containing img array and corresponding output value.")


if __name__=="__main__":
    generate_test_file()
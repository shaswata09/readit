import cv2
import numpy as np
import glob
import pickle


def generate_train_file():
    out_lst = ["0","1","2","3","4","5","6","7","8","9",
               "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
               "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    prcsd_img_list = (glob.glob("/home/leonardo/Desktop/readit_test/training_dataset/processed_img/*"))
    prcsd_img_file = "/home/leonardo/Desktop/readit_test/training_dataset/processed_img_file.pkl"
    train_img_list = list()

    for path in prcsd_img_list:
        imgpath = path
        out_val = str(path[66])
        img = cv2.imread(imgpath, 0)
        img = np.reshape(img, (1600, 1))

        img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')
        img_res = np.ndarray(shape=(62, 1), dtype=float, order='F')

        for i in range(img.shape[0]):
            img_arr[i] = (float(img[i] / 256))
        for i in range(62):
            img_res[i] = (float(0))
        out_indx = out_lst.index(out_val)
        img_res[out_indx] = float(1)
        img_tup = (img_arr, img_res)
        train_img_list.append(img_tup)


    with open(prcsd_img_file, 'wb') as f:
        pickle.dump(train_img_list, f)

    print(str(len(train_img_list))+" images training list is created.")
    print("The list contain tuples containing img array and corresponding output vector.")



if __name__ == "__main__":
    generate_train_file()

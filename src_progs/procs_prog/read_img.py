import cv2
import numpy as np
import time
import sys
sys.path.insert(0, '/home/leonardo/Desktop/readit_test/src_progs/procs_prog/')
from focus_extend import focus_extend
from read_network import read_network
from scaling_img import scale_image

def background_checking(img):
    img_arr = np.asarray(img)
    avg_intensity = int((int(img_arr[0][0]) + int(img_arr[0][39]) + int(img_arr[39][0]) + int(img_arr[39][39])) / 4)

    if avg_intensity > 127 :
        ret, new_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img=new_img
    return img


def feedforward(self, a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)
    return a


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def read_img(img_path):

    net = read_network()
    out_lst = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y", "Z",
               "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
               "u", "v", "w", "x", "y", "z"]

    print("Image is now processing...")

    img = focus_extend(img_path)

    pic_original = cv2.imread(img_path, 1)
    pic_original = cv2.resize(pic_original, (img.shape[1],40), interpolation=cv2.INTER_AREA)

    # img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)


    # seperating characters from the word into char_brk_indx
    pxl_sum_lst = []
    char_brk_indx = []
    for i1 in range(img.shape[1]):
        sum = 0
        for i2 in range(40) :
            sum = sum + img[i2][i1]
        pxl_sum_lst.append(sum)

    i1 = 0
    print(img.shape[1])
    while i1 < img.shape[1]:
        while i1 < img.shape[1] and pxl_sum_lst[i1] <= 500:
            i1 = i1 + 1
        i2 = i1
        while i2 < img.shape[1] and pxl_sum_lst[i2] > 500:
            i2 = i2 + 1
        char_brk_indx.append((i1, i2))
        i1 = i2

    print(char_brk_indx)

    for char_indx in char_brk_indx:
        if(char_indx[1]-char_indx[0]<5):
            char_brk_indx.remove((char_indx[0],char_indx[1]))

    print(char_brk_indx)


    # for j1 in range(gaps):
    #     if char_brk_indx[j1]-strt > 40:
    #         hid_wrd = int((char_brk_indx[j1] - strt)/40)
    #         for j2 in range(hid_wrd):
    #             char_brk_indx.append(pxl_sum_lst.index(min(pxl_sum_lst[strt+8 : strt+48]), strt+8, strt+48))
    #             strt = strt+35
    #     strt = char_brk_indx[j1]
    # char_brk_indx.sort()


    print(f"There are {len(char_brk_indx):<2} possible characters.")
    # print(char_brk_indx)


    # scan_arr = np.zeros((40, char_brk_indx[0] + 1), float)
    # for i1 in range(40):
    #     for j1 in range(scan_arr.shape[1]):
    #         scan_arr[i1][j1] = img[i1][j1]
    #
    # scan_arr = cv2.resize(scan_arr, (40, 40))
    # cv2.imshow("m",scan_arr)
    # cv2.imshow('Output_focused', img)
    # print(scan_arr.shape)
    # half_converted_img = np.reshape(scan_arr, (1600, 1))
    # img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')
    # for i in range(half_converted_img.shape[0]):
    #     img_arr[i] = (float(half_converted_img[i] / 256))
    # result_vector = feedforward(net, img_arr)
    # result_indx = np.argmax(result_vector)
    # result_prob = result_vector[result_indx]
    # result_prob = float(result_prob * 100)
    # out_val = out_lst[result_indx]
    # print(out_val)




    # reading seperated characters from char_brk_indx list
    out_word = ""
    out_prob = 0

    for i3 in range(len(char_brk_indx)):
        scan_arr = np.zeros((40, char_brk_indx[i3][1]-char_brk_indx[i3][0]), float)
        for i1 in range(40):
            for j1 in range(scan_arr.shape[1]):
                scan_arr[i1][j1] = img[i1][char_brk_indx[i3][0]+j1]
        scan_arr = cv2.resize(scan_arr, (40, 40))


        half_converted_img = np.reshape(scan_arr, (1600, 1))
        img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')
        for i in range(half_converted_img.shape[0]):
            img_arr[i] = (float(half_converted_img[i] / 256))
        result_vector = feedforward(net, img_arr)
        result_indx = np.argmax(result_vector)
        result_prob = result_vector[result_indx]
        result_prob = float(result_prob * 100)
        out_val = out_lst[result_indx]
        out_word = out_word + out_val
        out_prob = out_prob + result_prob

        cv2.imshow(f"Character_{i3}", scan_arr)


    out_prob = out_prob / len(out_word)

    log_path = "/home/leonardo/Desktop/readit_test/sys_log/log.txt"
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{img_path:<60}{out_word:<10}{out_prob:<25} %          {time.time():<20}")

    print(f"The image's possibly : {out_word:<15}</b> with a probability {out_prob:>20} %.")



    # while lft+39 < img.shape[1]:
    #     tmp_word_lst = []
    #     tmp_prob_lst = []
    #     for i2 in range(9, 40):
    #         scan_arr = np.zeros((40, i2+1), int)
    #         for i1 in range(40):
    #             for j1 in range(i2+1):
    #                 scan_arr[i1][j1] = img[i1][lft+j1]
    #
    #         temp_arr = np.resize(scan_arr, (40,40))
    #         half_converted_img = np.reshape(temp_arr, (1600, 1))
    #         img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')
    #         for i in range(half_converted_img.shape[0]):
    #             img_arr[i] = (float(half_converted_img[i] / 256))
    #         result_vector = feedforward(net, img_arr)
    #         result_indx = np.argmax(result_vector)
    #         result_prob = result_vector[result_indx]
    #         result_prob = float(result_prob * 100)
    #         out_val = out_lst[result_indx]
    #         tmp_word_lst.append(out_val)
    #         tmp_prob_lst.append(result_prob)
    #     out_word = out_word + tmp_word_lst[tmp_prob_lst.index(max(tmp_prob_lst))]
    #     out_prob = out_prob + max(tmp_prob_lst)
    #     lft = lft + tmp_prob_lst.index(max(tmp_prob_lst))+9
    #
    #
    # out_prob = out_prob / len (out_word)
    # print(f"The image's possibly word : {out_word:<15} with a probability {out_prob:>20} %.")

    # half_converted_img = np.reshape(img, (1600, 1))
    # img_arr = np.ndarray(shape=(1600, 1), dtype=float, order='F')
    #
    # for i in range(half_converted_img.shape[0]):
    #     img_arr[i] = (float(half_converted_img[i] / 256))
    #
    # result_vector = feedforward(net, img_arr)
    # result_indx= np.argmax(result_vector)
    # result_prob = result_vector[result_indx]
    # result_prob = float(result_prob * 100)
    # out_val = out_lst[result_indx]
    #
    # log_path = "/home/leonardo/Desktop/readit_test/sys_log/log.txt"
    # with open(log_path, 'a') as log_file:
    #     log_file.write(f"\n{img_path:<60}{out_val:<10}{result_prob:<25} %          {time.time():<20}")
    # print(f"The image's possibly character : {out_val:<5} with a probability {result_prob:>20} %.")

    cv2.imshow('Output_focused', img)
    cv2.imshow('Output', pic_original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img_path = "/home/leonardo/Desktop/readIT_livetest/shaswata.png"
    # img_path = str(sys.argv[1])
    read_img(img_path)

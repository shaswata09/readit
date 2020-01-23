import cv2
import numpy as np


def rotate_img_90(img):
    dim=img.shape[0]-1
    temp_img = np.zeros((img.shape[1], img.shape[0]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp_img[j][dim-i] = img[i][j]

    return temp_img


def left_scale(img):

    left=10
    # left
    for i in range(0,img.shape[1]):
        sum = 0
        for j in range(img.shape[0]):
            sum+=img[j][i]
        if sum>765 :
            left=i
            break

    temp_img=np.zeros((img.shape[0],(img.shape[1]-left+10)))
    for i in range(10,temp_img.shape[1]):
        for j in range(temp_img.shape[0]):
            temp_img[j][i] = img[j][left + i - 10]

    return temp_img


def background_checking(img):
    img_arr = np.asarray(img)
    avg_intensity = int(img_arr[0][0]) + int(img_arr[0][img_arr.shape[1]-1]) + int(img_arr[img_arr.shape[0]-1][0]) + int(img_arr[img_arr.shape[0]-1][img_arr.shape[1]-1]) / 4

    if avg_intensity > 127 :
        ret, new_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img = new_img
    return img


def scale_image(img):
    # left
    img = left_scale(img)

    # bottom
    img = rotate_img_90(img)
    img = left_scale(img)

    # right
    img = rotate_img_90(img)
    img = left_scale(img)

    #top
    img = rotate_img_90(img)
    img = left_scale(img)

    # get original aligned image
    img = rotate_img_90(img)
    return img


def focus_img(img_path):

    pic = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(pic, (5, 5), 0)
    ret3, thresh_output = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = background_checking(thresh_output)
    img = scale_image(img)

    return img


if __name__ == '__main__':
    img_path = "/home/leonardo/Desktop/readIT_livetest/l.png"
    pic = cv2.imread(img_path, 1)
    cv2.imshow("Original input", pic)
    img = focus_img(img_path)
    cv2.imshow("scaled output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
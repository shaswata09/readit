import cv2
import sys
sys.path.insert(0, '/home/leonardo/Desktop/readit_test/src_progs/procs_prog/')
from scaling_img import background_checking
from scaling_img import scale_image

def focus_img(pic):

    blur = cv2.GaussianBlur(pic, (5, 5), 0)
    ret3, thresh_output = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img = background_checking(thresh_output)
    img = scale_image(img)

    return img



def resize_img(img):

    ratio = 40/img.shape[0]
    width = int(img.shape[1] * ratio )
    height = int(img.shape[0] * ratio)
    dim = (width, height)

    img_org = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img_org

def focus_extend(img_path):

    img = cv2.imread(img_path, 0)

    sum = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(i==0):
                sum += img[i][j]
            elif(j==0):
                sum+=img[i][j];
            elif(i== img.shape[0] - 1):
                sum+=img[i][j];
            elif(j== img.shape[1] - 1):
                sum+=img[i][j]

    sum = sum - img[0][0] - img[img.shape[0] - 1][0] - img[img.shape[0] - 1][img.shape[1] - 1] - img[1][img.shape[1] - 1]

    avg_intensity = sum/(2*(img.shape[0]+img.shape[1]))

   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (abs(avg_intensity - img[i][j]) > 60):
                img[i][j] = 255
            else:
                img[i][j]=0

    img = focus_img(img)

    img = resize_img(img)


    return img


if __name__ == '__main__':
    img_path = "/home/leonardo/Desktop/readIT_livetest/j.jpg"
    img_org = cv2.imread(img_path, 0)


    cv2.imshow("Grayscale Image", img_org)

    img = focus_img(img_org)

    cv2.imshow("Modified Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

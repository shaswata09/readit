import cv2
import numpy as np
from PIL import Image
import glob



def background_checking(img):
    img_arr = np.asarray(img)
    #cv2.imshow('Normal', img)
    #print(img_arr)

    if img_arr[0][0] == 255 or img_arr[0][39] == 255 or img_arr[39][0] == 255 or img_arr[39][39] == 255:
        ret, new_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img_arr2 = np.asarray(new_img)
        #cv2.imshow('INV', thresh2)
        #print(img_arr2)
        img = new_img

    return img




def ImageProcessing1(input_path, output_path) :

    # Image Reading Step
    pic = cv2.imread(input_path, 0)



    #cv2.imwrite('C:/Users/Sourav Dutta/Documents/Python Project/Output/TEST13.png',pic)
    # Display Original Image
    #cv2.imshow('Original Image',pic)



    #Resizing Image in 40 x 40
    pic2 = cv2.resize(pic, (40,40), interpolation = cv2.INTER_AREA)



    # Thresholding
    #thresh_output = cv2.adaptiveThreshold(pic2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Thresholding ( GAUSSIANBLUR + BINARY THRESHOLDING )
    blur = cv2.GaussianBlur(pic2, (5,5), 0)
    ret3,thresh_output = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    #checking image background
    img = background_checking(thresh_output)


    # Saving the Image
    cv2.imwrite(output_path,img)


    #cv2.imshow('GAUSS', thresh_output2)



    # For only OSTU adaptive thresholding

    # ret2,thresh_output = cv2.threshold(pic2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imwrite('C:/Users/Sourav Dutta/Documents/Python Project/Output/TEST11_OSTU.png', thresh_output)
    #cv2.imshow('OSTU', thresh_output)


    #print(pic2)

    #cv2.waitKey(0)





def main():

    file_path = glob.glob("/home/sourav/OCR FILES/Input/original_img_2/Images/*")
    output_path = "/home/sourav/OCR FILES/Output/"
    img_number = 1
    target = 'Images'

    for i in file_path:
                # Editing the file path for the REPLACING backslash to frontslash [FOR WINDOWS]
                #indx = i.find(target)
                #indx = indx + 11

                # for maintaining the same name as the input files
                #output_file_name = i[indx:]
                #output_path_new = output_path + output_file_name


                #new_str = i[0:indx - 1]
                #new_str = new_str + '/' + i[indx:]
                #i = new_str




        # [FOR LINUX]      for maintaining the same name as the input files
        indx = i.find(target)
        indx = indx + 7
        output_file_name = i[indx:]
        output_path_new = output_path + output_file_name


        ImageProcessing1(i,output_path_new)

        print(i)
        print(output_path_new)
        img_number += 1


if __name__ == '__main__':
    main()

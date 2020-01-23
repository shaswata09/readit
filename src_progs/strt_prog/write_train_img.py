import cv2
import glob
from src_progs.procs_prog.scaling_img import focus_img

def write_img(input_dir, output_dir):

    file_path = glob.glob(input_dir)
    img_number = 1
    target = 'img'

    for i in file_path:
        # [FOR LINUX]      for maintaining the same name as the input files
        indx = i.find(target)
        indx = indx + 4
        output_file_name = i[indx:]
        output_path_new = output_dir + output_file_name

        img = focus_img(i)
        img = cv2.resize(img,(40,40), interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path_new, img)

        img_number += 1


if __name__ == '__main__':

    output_dir = "/home/leonardo/Desktop/readit_test/training_dataset/processed_img/"
    input_dir = "/home/leonardo/Desktop/readit_test/training_dataset/original_img/*"

    write_img(input_dir, output_dir)
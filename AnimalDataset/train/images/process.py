import os
import cv2


list_dirs = 'AnimalDataset/train/images/image'

file_list = os.listdir(list_dirs)

for i, file in enumerate(file_list):
    img_path = os.path.join(list_dirs, file)
    orig_image = cv2.imread(img_path)

    new_img_name = os.path.join(list_dirs, '{}.jpg'.format(i+12))
    cv2.imwrite(new_img_name, orig_image)
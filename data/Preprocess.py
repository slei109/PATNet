import numpy as np
import glob as glob
import cv2
import os,shutil
from PIL import Image

### This records the main operations to preprocess the data in Deepglobe Dataset

################################################################
# step 1: cut the big image and mask into smaller one
################################################################


# for f in os.listdir(path_folder):
#
#     path = YOUR_PATH_TO_DATA + f.strip()
#     img = cv2.imread(path)
#     hei = img.shape[0]
#     wid = img.shape[1]
#
#     # get 6 roi
#     num = 6
#     for i in range(0, num):
#         for j in range(0, num):
#             print(i)
#             hei_0 = int(i * hei / num)
#             hei_1 = int((i + 1) * hei / num)
#             wid_0 = int(j * wid / num)
#             wid_1 = int((j + 1) * wid / num)
#             roiImg = img[hei_0:hei_1, wid_0:wid_1]
#             if f.endswith('.jpg'):
#                 path = YOUR_PATH_TO_IMAGE + f.strip()[0:-4] + "_" + str(i) + str(j) + ".jpg"
#                 cv2.imwrite(path, roiImg)
#             else:
#                 path = YOUR_PATH_TO_LABEL + f.strip()[0:-4] + "_" + str(i) + str(j) + ".png"
#                 cv2.imwrite(path, roiImg)



############################################################################
# step 2: filter single color mask and foreground percent small than 0.048
############################################################################


# path_folder = os.path.expanduser(YOUR_PATH_TO_DATA)
#
# for f in os.listdir(path_folder):
#     path = YOUR_PATH_TO_DATA + f.strip()
#     img = Image.open(path)
#     colors = img.getcolors()
#     percent_min = 1
#     if len(colors) > 1:
#         for num in colors:
#             percent = num[0] / 166464
#             if percent < percent_min:
#                 percent_min = percent
#         if percent_min > 0.048:
#             path = YOUR_PATH_TO_FILTER_DATA + f.strip()
#             img.save(path)


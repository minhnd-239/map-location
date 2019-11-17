import cv2
import numpy as np
import glob
import os
from dask.array.tests.test_numpy_compat import dtype
# alist =[]
# count = 0
list_matches = []
# data = "Image/test"
# data_path = os.path.join(data, '*.jpg')
# file = glob.glob("Image/test/*.jpg")
# for img in file:
#     train = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  
#     orb = cv2.ORB_create()
#     kp2, des2 = orb.detectAndCompute(train, None)
#     file_name = "IMG_" + str(count) + ".txt"
#     with open ("Image/15/" +file_name, "a") as f:
# #         f.write(str(kp2))
# #         f.write(";")
#         if des2 is not None:
#             for i in des2:
#                 f.write(str(i))
#             f.close()
#     count+=1
def compare_feature(des):
    data = "Image/15"
    data_path = os.path.join(data, '*.txt')
    file_name = glob.glob(data_path)
    temp =[]
    for file in file_name:
        with open (file, "r") as file:
            contents = file.read()
#             print(contents)
            temp = np.array((contents))
            print(contents)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#             print(contents)
            matches = bf.match(des, temp)
#             print(matches)
            list_matches.append(matches)
        file.close()
#             contents = txt.split(";")
            
            
            
    
  
    
print('ok')
query = cv2.imread("IMG_0448 011.jpg", cv2.IMREAD_GRAYSCALE)
# # train = cv2.imread("IMG_0448 001.jpg", cv2.IMREAD_GRAYSCALE)
# # query = cv2.resize(query, (500,500))
# # train = cv2.resize(train, (500,500))
orb = cv2.ORB_create()
#    
kp1, des1 = orb.detectAndCompute(query, None)
compare_feature(des1)
print(list_matches)
# # kp2, des2 = orb.detectAndCompute(train, None)
# print(des2)
# for i in des2:
#     print(str(i))
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# for m in matches:
#     print(m.distance)
# print(len(matches))
#  
# matching_result = cv2.drawMatches(query, kp1, train, kp2, matches[:20], None, flags = 2)
# cv2.imshow("query", query)
# cv2.imshow("train", train)
# cv2.imshow("Matching", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


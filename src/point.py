import cv2
import numpy as np
import glob
import os
from dask.array.tests.test_numpy_compat import dtype
# alist =[]
count = 1
list_matches = []
# 
# for filename in glob.glob('Image/test/*.jpg'):
# #     print(filename[-7:-4])
#          
#     train = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# #     cv2.imwrite("Image/10/" + "%s.jpg" %filename[-7:-4], train)
# #     cv2.imshow("train", train)
#     #cv2.waitKey(0)
#     orb = cv2.ORB_create()
#     kp2, des2 = orb.detectAndCompute(train, None)
#     file_name = "Image/15/IMG_" + str(count)
#     if des2 is not None:
#         np.save(file_name, des2)
#     count+=1


def compare_feature(des):
  
    for filename in glob.glob('Image/15/*.npy'):
        print(filename)
        contents = np.load(filename) # load
#         print(contents)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #             print(contents)
        matches = bf.match(des, contents)
#         print(len(matches))
        list_matches.append(len(matches))


            
            
    
  
    
print('ok')
query = cv2.imread("Image/10/010.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("query", query)
# train = cv2.imread("Image/10/013.jpg", cv2.IMREAD_GRAYSCALE)
# query = cv2.resize(query, (500,500))
# train = cv2.resize(train, (500,500))
orb = cv2.ORB_create()
#  
kp1, des1 = orb.detectAndCompute(query, None)
# kp2, des2 = orb.detectAndCompute(train, None)
compare_feature(des1)
print(list_matches)
print(max(list_matches))
# kp2, des2 = orb.detectAndCompute(train, None)
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


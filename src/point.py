import numpy as np
import cv2
import glob
count = 0
list_matches = []
def train():
    for filename in glob.glob('data_test/*.jpg'):
    # #     print(filename[-7:-4])
    #          
        train = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #     cv2.imwrite("Image/10/" + "%s.jpg" %filename[-7:-4], train)
    #     cv2.imshow("train", train)
        #cv2.waitKey(0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des2 = sift.detectAndCompute(train,None)
        file_name = "data_train/IMG_" + str(count)
        if des2 is not None:
            np.save(file_name, des2)
        count+=1
    
    
def compare_feature(des):
  
    for filename in glob.glob('data_train/*.npy'):
        print(filename)
        contents = np.load(filename) # load
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(des, contents)
        list_matches.append(len(matches))
# print('ok')
train = cv2.imread('data_test/001.jpg',cv2.IMREAD_GRAYSCALE)
query = cv2.imread('data_test/001.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
kp_train, des_train = sift.detectAndCompute(train,None)
kp_query, des_query = sift.detectAndCompute(train,None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des_train, des_query)
print(len(matches))
img3 = cv2.drawMatches(train, kp_train, query, kp_query, matches, query, flags=2)
cv2.imshow("aaa", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
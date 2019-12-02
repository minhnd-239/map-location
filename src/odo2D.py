import numpy as np
import cv2
import timeit
import glob
import re
def getAbsoluteScale(f, frame_id):
      x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
      x    , y    , z     = f[frame_id][3], f[frame_id][7], f[frame_id][11]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale
      
def featureTracking(img_1, img_2, p1):

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2


def featureDetection():
#     thresh = dict(threshold=25, nonmaxSuppression=True);
#      fast = cv2.FastFeatureDetector_create(**thresh)
#     return fast
#    orb = cv2.ORB_create(edgeThreshold=25, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
#     sift=cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
#     return sift
    return orb


def getImages(i):
    return cv2.imread('D:/Eclipse/pyslam/data_test/{0:03d}.jpg'.format(i),0)

def getK():
    return   np.array([[9434.76, 0, 364.34],
              [0, 9272.73, 2],
              [0, 0, 1]])


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts   
def compare_feature(des):
    coor = []
    for filename in sorted(glob.glob('data_train/*.npy'), key=numericalSort):
#         print(filename)
        contents = np.load(filename) # load
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #             print(contents)
        matches = bf.match(des, contents)
        if len(matches) > 450:
            print(len(matches))
#             list_matches.append(len(matches))
            coor = filename[-11:-4].split("-")
            list_matches.append(coor[0])
            list_matches.append(coor[1])
            break
        
        
def train(des, x, y):
    file_name = "data_train/IMG" + str(x) + "-" + str(y)
    if des is not None:
        np.save(file_name, des)
#initialization
#sground_truth =getTruePose()
cap = cv2.VideoCapture('video.mp4')
f, img_1 = cap.read()
f, img_2 = cap.read()

if len(img_1) == 3:
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
else:
    gray_1 = img_1
    gray_2 = img_2

#find the detector
detector = featureDetection()
kp1, des1      = detector.detectAndCompute(img_1, None)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = featureTracking(gray_1, gray_2, p1)

#Camera parameters
fc = 600
pp = (320.087, 240.3019)
K  = getK()

E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);

#initialize some parameters
MAX_FRAME       = 300
MIN_NUM_FEAT  = 200

preFeature = p2
preImage   = gray_2

R_f = R
t_f = t



traj = np.zeros((1200, 1200, 3), dtype=np.uint8)

maxError = 0
count = 0
#play image sequences
img=0
c=0
list_matches = []
contents= []
numbers = re.compile(r'(\d+)')



bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    
while(True):
    start = timeit.default_timer()
# for numFrame in range(2, MAX_FRAME):
    f, temp1 = cap.read()
    if count ==0:
        img= temp1
    count+=1
    if count == 5:
        count =0
        temp = np.zeros((480, 1280, 3), dtype=np.uint8)
        if (len(preFeature) < MIN_NUM_FEAT):
            feature, des2   = detector.detectAndCompute(preImage, None)
            preFeature = np.array([ele.pt for ele in feature],dtype='float32')
        curImage_c = img    
        if len(curImage_c) == 3:
              curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)
        else:
              curImage = curImage_c
        
        kp1, des3 = detector.detectAndCompute(curImage, None);
        
        compare_feature(des3)
        
        print(list_matches)
        list_matches= []
            
        img1=cv2.drawKeypoints(curImage,kp1,curImage_c)
        #cv2.imshow("ve",img1)
        #print(len(kp1))
        preFeature, curFeature = featureTracking(preImage, curImage, preFeature)
        E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0); 
        _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp);
        t_f = t_f + 1*R_f.dot(t)    
        R_f = R.dot(R_f)   
        preImage = curImage
        preFeature = curFeature
        
    
        ####Visualization of the result
        draw_x, draw_y = int(t_f[0]) + 300, int(t_f[2]) + 300;
        print(draw_x)
        print(draw_y)
        #save description of frame to database
#         train(des3, draw_x, draw_y)
        cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2);    
        cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED);
        text = "khoang cach so voi land mark: x ={0:02f}m y = {1:02f}m".format(float(500-draw_x), float(500-draw_y));
        cv2.drawKeypoints(curImage, kp1, temp)
#         cv2.imshow('temp', temp)
        cv2.imshow('image', curImage_c)
        cv2.imshow( "Trajectory", traj )
        stop = timeit.default_timer()
        print(stop - start)
   
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

print('Maximum Error: ', maxError)
cv2.imwrite('map.png', traj);

cv2.destroyAllWindows()

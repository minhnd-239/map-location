import numpy as np
import cv2
import timeit

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
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast


def getImages(i):
    return cv2.imread('D:/Eclipse/pyslam/data_test/{0:03d}.jpg'.format(i),0)

def getK():
    return   np.array([[9434.76, 0, 364.34],
              [0, 9272.73, 2],
              [0, 0, 1]])


#initialization
#sground_truth =getTruePose()

img_1 = getImages(1)
img_2 = getImages(1)

if len(img_1) == 3:
	gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
	gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
else:
	gray_1 = img_1
	gray_2 = img_2

#find the detector
detector = featureDetection()
kp1      = detector.detect(img_1)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = featureTracking(gray_1, gray_2, p1)

#Camera parameters
fc = 2000
pp = (320.087, 240.3019)
K  = getK()

E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);

#initialize some parameters
MAX_FRAME 	  = 500
MIN_NUM_FEAT  = 500

preFeature = p2
preImage   = gray_2

R_f = R
t_f = t

start = timeit.default_timer()

traj = np.zeros((1200, 1200, 3), dtype=np.uint8)

maxError = 0

#play image sequences
for numFrame in range(2, MAX_FRAME):
    temp = np.zeros((480, 1280, 3), dtype=np.uint8)
    print(numFrame)

    if (len(preFeature) < MIN_NUM_FEAT):
        feature   = detector.detect(preImage)
        preFeature = np.array([ele.pt for ele in feature],dtype='float32')

    #filename = imgs.format(numFrame)
    #print(filename)
    curImage_c = getImages(numFrame)

    if len(curImage_c) == 3:
          curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)
    else:
          curImage = curImage_c
    
    kp1 = detector.detect(curImage);
    preFeature, curFeature = featureTracking(preImage, curImage, preFeature)
    E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0); 
    _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp);

#    truth_x, truth_y, truth_z, absolute_scale = getAbsoluteScale(ground_truth, numFrame)

#    if absolute_scale > 0.1:
    t_f = t_f + 1*R_f.dot(t)
    R_f = R.dot(R_f)

    preImage = curImage
    preFeature = curFeature
    

    ####Visualization of the result
    draw_x, draw_y = int(t_f[0]) + 300, int(t_f[2]) + 300;
#    draw_tx, draw_ty = int(truth_x) + 300, int(truth_z) + 100

#    curError = np.sqrt((t_f[0]-truth_x)**2 + (t_f[1]-truth_y)**2 + (t_f[2]-truth_z)**2)
 #   print('Current Error: ', curError)
#    if (curError > maxError):
#        maxError = curError
    cv2.circle(traj, (500, 500) ,1, (255,0,0), 2);  # dong nay dat land mark
    cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2);
#    cv2.circle(traj, (draw_tx, draw_ty) ,1, (255,0,0), 2);

    cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED);
    text = "khoang cach so voi land mark: x ={0:02f}m y = {1:02f}m".format(float(500-draw_x), float(500-draw_y));
    #cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8);
    cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8);
    cv2.drawKeypoints(curImage, kp1, temp)
    cv2.imshow('temp', temp)
    cv2.imshow('image', curImage_c)
    cv2.imshow( "Trajectory", traj );
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
  	#time.sleep(1)
# k = cv2.waitKey(0) & 0xFF
# if k == 27:
print('Maximum Error: ', maxError)
cv2.imwrite('map.png', traj);
stop = timeit.default_timer()
print(stop - start)
cv2.destroyAllWindows()

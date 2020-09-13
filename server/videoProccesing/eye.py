import cv2
import numpy
from os.path import join
import dlib

import numpy as np


def landmarks_to_np(landmarks, dtype="int"):
   
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_centers(img, landmarks):
    
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) #画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) #高斯模糊

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #y方向sobel边缘检测
    sobel_y = cv2.convertScaleAbs(sobel_y) #转换回uint8类型
    

    edgeness = sobel_y #边缘强度矩阵
    
    #Otsu二值化
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #计算特征长度
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] #提取ROI
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#计算评价值
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#计算评价值
    measure = measure_1*0.3 + measure_2*0.7
    
    
    print(measure)
    
    #根据评价值和阈值的关系确定判别值
    if measure > 0.3:#阈值可调，经测试在0.15左右
        judge = True
    else:
        judge = False
    
    return judge 





def show_image_with_data(frame, blinks, irises, err=None):
    """
    Helper function to draw points on eyes and display frame
    :param frame: image to draw on
    :param blinks: number of blinks
    :param irises: array of points with coordinates of irises
    :param err: for displaying current error in Lucas-Kanade tracker
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'morganie: ' + str(blinks), (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    return frame
    #cv2.imshow('Eyeris detector', frame)


class ImageSource:
    """
    Returns frames from camera
    """


    def get_current_frame(self,frame):
        gray=False
        frame = cv2.flip(frame, 1)
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




class CascadeClassifier:
    """
    This classifier is trained by default in OpenCV
    """
    def __init__(self, glasses=True):
        if glasses:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))

    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        irises = []

        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

        return numpy.array(irises)


class LucasKanadeTracker:
    """
    Lucaas-Kanade tracker used for minimizing cpu usage and blinks counter
    """
    def __init__(self, blink_threshold=9):
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = blink_threshold

    def track(self, old_gray, gray, irises, blinks, blink_in_previous):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **self.lk_params)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            blink_in_previous = False
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
            lost_track = True
            if not blink_in_previous:
                blinks += 1
                blink_in_previous = True
        else:
            blink_in_previous = False
            irises = []
            for w, h in p1:
                irises.append([w, h])
            irises = numpy.array(irises)
        return irises, blinks, blink_in_previous, lost_track


class EyerisDetector:
    """
    Main class which use image source, classifier and tracker to estimate iris postion
    Algorithm used in detector is designed for one person (with two eyes)
    It can detect more than two eyes, but it tracks only two
    """
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.irises = []
        self.blink_in_previous = False
        self.blinks = 0
        predictor_path = "videoProccesing/data/shape_predictor_5_face_landmarks.dat"#人脸关键点训练数据路径
        self.detector = dlib.get_frontal_face_detector()#人脸检测器detector
        self.predictor = dlib.shape_predictor(predictor_path)#人脸关键点检测器predictor

    def run(self,frame):
        
        if True:
            frame = self.image_source.get_current_frame(frame)
            judge = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                if lost_track:
                    self.irises = self.classifier.get_irises_location(gray)
            else:  # cannot track for some reason -> find irises
                self.irises = self.classifier.get_irises_location(gray)
           
            
            frame = show_image_with_data(frame, self.blinks, self.irises)
            
            #-------------------2-----------------
            img = frame
            rects = self.detector(gray, 1)
    
    
            for i, rect in enumerate(rects):
     
              x_face = rect.left()
              y_face = rect.top()
              w_face = rect.right() - x_face
              h_face = rect.bottom() - y_face
        
       
              cv2.rectangle(img, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
              #cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
                 
              landmarks = self.predictor(gray, rect)
              landmarks = landmarks_to_np(landmarks)
              for (x, y) in landmarks:
                          cv2.circle(img, (x, y), 2, (0, 255, 255), -1)

      
              LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        
     
              aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
              
        
      
              judge = judge_eyeglass(aligned_face)
              '''if judge == True:
                                        cv2.putText(img, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
              else:
                                        cv2.putText(img, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)'''
            rez = {}
            rez['eye']=self.irises
            rez['glasses']=judge
            rez['eye_l_centre']=LEFT_EYE_CENTER
            rez['eye_r_centre']=RIGHT_EYE_CENTER
            rez['label'] = landmarks
            
            print(rez)
            
            old_gray = gray.copy()
            return img

        
eyeris_detector = EyerisDetector(image_source=ImageSource(), classifier=CascadeClassifier(),
                                 tracker=LucasKanadeTracker())
while True:
    





import numpy as np
import cv2
from lib.kalman_filter import Kalman_filter

cap = cv2.VideoCapture(0)

def findMaxmiumArea(target):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(target)
    maxHeight = 0
    indexHeight = 0
    maxwidth = 0
    index = 0
    for i in range(1,nlabels):
        if stats[i, cv2.CC_STAT_HEIGHT] > maxHeight:
            maxHeight = stats[i, cv2.CC_STAT_HEIGHT]
            index = i
            if stats[index, cv2.CC_STAT_WIDTH] > maxwidth:
                maxwidth = stats[index, cv2.CC_STAT_WIDTH]
                indexHeight = i
    return stats, centroids, indexHeight


def detect_object(frame):
    h, w, c = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 7)
    hsv = cv2.GaussianBlur(hsv,(7,7),0)

    upper_blue = np.array([130,255,255])
    lower_blue = np.array([90,100,100])

    target = cv2.inRange(hsv, lower_blue, upper_blue)

    hi, wi = target.shape

    x, y, left, top, width, height = 0, 0, 0, 0, 0, 0

    stats, centroids, indexHeight = findMaxmiumArea(target)

    left = stats[indexHeight, cv2.CC_STAT_LEFT]
    top = stats[indexHeight, cv2.CC_STAT_TOP]
    width = stats[indexHeight, cv2.CC_STAT_WIDTH]
    height = stats[indexHeight, cv2.CC_STAT_HEIGHT]
    x = int(centroids[indexHeight, 0])
    y = int(centroids[indexHeight, 1])

    crop_img = target[top:top+height, left:left+width]

    area = height * width
    if area > 0 and area < 50000:
        u = 'target'
        buff = 10
        cv2.putText(frame, str(u), (left-buff, top-buff), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.circle(frame, (x, y), 7, (0,0,255), -1)
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0,0,255), 2)
        return frame, x, y
    else:
        return frame, 0, 0


def open_video(cap):
    global kf
    while True:
        _, frame = cap.read()
        if _:
            h, w, c = frame.shape
            frame, x, y = detect_object(frame)
            
            if x != 0:

                current_mes = np.array([[np.float32(x)],[np.float32(y)]])
                
                preX, preY = kf.predict()
                X, P = kf.update(np.array([x, y]))
                
                cpx, cpy = X[0,0],X[1,0]

                cv2.circle(frame, (int(cpx),int(cpy)), 4, (0, 255, 0), -1)

            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
kf = Kalman_filter(time=0.001)
open_video(cap)
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
cap.release()
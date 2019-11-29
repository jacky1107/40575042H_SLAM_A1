import numpy as np
import cv2

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
    if area > 300 and area < 50000:
        u = 'target'
        buff = 10
        cv2.putText(frame, str(u), (left-buff, top-buff), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.circle(frame, (x, y), 7, (0,0,255), -1)
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0,0,255), 2)
        return frame, x, y
    else:
        return frame, 0, 0


def open_video(cap):
    global current_mes, last_mes, current_pre, last_pre
    while True:
        _, frame = cap.read()
        if _:
            h, w, c = frame.shape
            frame, x, y = detect_object(frame)
            last_pre = current_pre
            last_mes = current_mes
            
            if x != 0:

                current_mes = np.array([[np.float32(x)],[np.float32(y)]])

                kalman.correct(current_mes)
                current_pre = kalman.predict()

                lmx, lmy = last_mes[0],last_mes[1]
                lpx, lpy = last_pre[0],last_pre[1]
                cmx, cmy = current_mes[0],current_mes[1]    
                cpx, cpy = current_pre[0],current_pre[1]
                cv2.circle(frame, (cpx,cpy), 3, (0, 255, 0), -1)
#                 cv2.line(frame, (lmx,lmy),(cmx,cmy),(0,200,0), 4)
#                 cv2.line(frame, (lpx,lpy),(cpx,cpy),(0,0,200), 4)

            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
last_mes = current_mes = np.array((2,1),np.float32)
last_pre = current_pre = np.array((2,1),np.float32)
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
open_video(cap)
cap.release()
cv2.destroyAllWindows()
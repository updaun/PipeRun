import cv2
import numpy as np
import time
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import modules.HolisticModule as hm
detector = hm.HolisticDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# video test
# cap = cv2.VideoCapture("demo/lunge.mp4")

count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)

    img = detector.findHolistic(img, draw=False)
    pose_lmList = detector.findPoseLandmark(img, draw=False)
    # print(lmList)
    if len(pose_lmList) != 0:
        if pose_lmList[11][3] > pose_lmList[12][3]:
            rightleg_angle = 185 - detector.findAngle(img, 28, 26, 24, draw=True)
            leftleg_angle = 185 - detector.findAngle(img, 27, 25, 23, draw=True)
            total_angle = rightleg_angle + leftleg_angle

            x, y = pose_lmList[26][1:3] 
            x_core, y_core = pose_lmList[25][1:3]

            cv2.putText(img, str(int(rightleg_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(img, str(int(leftleg_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            

            # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            per = np.interp(total_angle, (80, 175), (0, 100))
            bar = np.interp(total_angle, (80, 175), (450, 100))
        
        else:
            rightleg_angle = detector.findAngle(img, 28, 26, 24, draw=True) - 170
            leftleg_angle = detector.findAngle(img, 27, 25, 23, draw=True) - 170
            total_angle = rightleg_angle + leftleg_angle

            x, y = pose_lmList[26][1:3] 
            x_core, y_core = pose_lmList[25][1:3]

            cv2.putText(img, str(int(rightleg_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(img, str(int(leftleg_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            

            # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            per = np.interp(total_angle, (80, 175), (0, 100))
            bar = np.interp(total_angle, (80, 175), (450, 100))
        
        # Check for the curls
        color = (255,0,255)
        if per == 100:
            color = (0,255,0)        
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0,255,0)        
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw bar
        cv2.rectangle(img, (550, 100), (590, 450), color, 3)
        cv2.rectangle(img, (550, int(bar)), (590, 450), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (540, 80),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)        

        # Draw curl count
        #cv2.putText(img, f'{count}', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)
        cv2.rectangle(img, (0, 300), (150, 480), (0, 255, 0), cv2.FILLED)
        if count < 10:
            cv2.putText(img, str(int(count)), (40, 420),
                        cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 12)
        else:
            cv2.putText(img, str(int(count)), (0, 420),
                        cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 12)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

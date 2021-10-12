import cv2
import numpy as np
import time
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import modules.HolisticModule as hm
import modules.SegmentationModule as sm

detector = hm.HolisticDetector()
bg_filter = sm.SegmentationFilter()


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# video test
# cap = cv2.VideoCapture("demo/squat_test.mp4")

count = 0
dir = 0
pTime = 0

bg_color = (192, 192, 192)
bg_image_path = 'images/_gym.jpg'

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    seg = img.copy()

    seg = bg_filter.Image(seg, img_path=bg_image_path)
    seg = cv2.resize(seg, (640, 480))

    img = detector.findHolistic(img, draw=False)
    pose_lmList = detector.findPoseLandmark(img, draw=False)

    # print(lmList)
    if len(pose_lmList) != 0:
        right_hand_x = pose_lmList[16][1]
        left_hand_x = pose_lmList[15][1]
        right_shoulder_x = pose_lmList[12][1]
        left_shoulder_x = pose_lmList[11][1]

        if right_hand_x < left_shoulder_x and right_shoulder_x < left_hand_x:
            if pose_lmList[11][1] > pose_lmList[12][1]:
                left_angle = 185 - detector.findAngle(seg, 24, 26, 28, draw=True)
                left_core_angle = 185 - detector.findAngle(seg, 26, 24, 12, draw=True)

                x, y = pose_lmList[26][1:3] 
                x_core, y_core = pose_lmList[24][1:3]

                cv2.putText(seg, str(int(left_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(seg, str(int(left_core_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                right_angle = 190 - detector.findAngle(seg, 27, 25, 23, draw=True)
                right_core_angle = 190 - detector.findAngle(seg, 11, 23, 25, draw=True)
                

                x, y = pose_lmList[25][1:3]
                x_core, y_core = pose_lmList[23][1:3]

                cv2.putText(seg, str(int(right_angle)), (x+40,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(seg, str(int(right_core_angle)), (x_core+40,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                total_angle = left_angle + left_core_angle + right_angle + right_core_angle
                # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                per = np.interp(total_angle, (10, 300), (0, 100))
                bar = np.interp(total_angle, (10, 300), (450, 100))
                
        # if pose_lmList[11][3] > pose_lmList[12][3]:
        elif right_hand_x > left_shoulder_x and right_shoulder_x < left_hand_x:
            angle = 185 - detector.findAngle(seg, 28, 26, 24, draw=True)
            core_angle = 185 - detector.findAngle(seg, 12, 24, 26, draw=True)
            total_angle = angle + core_angle

            x, y = pose_lmList[26][1:3] 
            x_core, y_core = pose_lmList[24][1:3]

            cv2.putText(seg, str(int(angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(seg, str(int(core_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            per = np.interp(total_angle, (10, 175), (0, 100))
            bar = np.interp(total_angle, (10, 175), (450, 100))
        
        elif right_hand_x < left_shoulder_x and right_shoulder_x > left_hand_x:
            angle = 190 - detector.findAngle(seg, 23, 25, 27, draw=True)
            core_angle = 190 - detector.findAngle(seg, 25, 23, 11, draw=True)
            total_angle = angle + core_angle

            x, y = pose_lmList[25][1:3]
            x_core, y_core = pose_lmList[23][1:3]

            cv2.putText(seg, str(int(angle)), (x+40,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(seg, str(int(core_angle)), (x_core+40,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            per = np.interp(total_angle, (10, 175), (0, 100))
            bar = np.interp(total_angle, (10, 175), (450, 100))

        # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        
        
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
        cv2.rectangle(seg, (550, 100), (590, 450), color, 3)
        cv2.rectangle(seg, (550, int(bar)), (590, 450), color, cv2.FILLED)
        cv2.putText(seg, f'{int(per)}%', (540, 80),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)        

        # Draw curl count
        #cv2.putText(img, f'{count}', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)
        cv2.rectangle(seg, (0, 300), (150, 480), (0, 255, 0), cv2.FILLED)
        if count < 10:
            cv2.putText(seg, str(int(count)), (40, 420),
                        cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 12)
        else:
            cv2.putText(seg, str(int(count)), (0, 420),
                        cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 12)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    cv2.imshow("Image", seg)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

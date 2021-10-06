import cv2
import numpy as np
import time
import mediapipe as mp
import modules.HolisticModule as hm

detector = hm.HolisticDetector()

mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
dir = 0
pTime = 0

bg_image = cv2.imread('images/_gym.jpg')


with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
    
    while True:
        success, img = cap.read()

        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        results = selfie_segmentation.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1

        img = np.where(condition, img, bg_image)
        img = cv2.resize(img, (640, 480))

        img = detector.findHolistic(img, draw=False)
        pose_lmList = detector.findPoseLandmark(img, draw=False)
        # print(lmList)
        if len(pose_lmList) != 0:
            if pose_lmList[11][3] > pose_lmList[12][3]:
                angle = 185 - detector.findAngle(img, 28, 26, 24)
                x, y = pose_lmList[26][1:3] 
                cv2.putText(img, str(int(angle)), (x-100,y+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            else:
                angle = 185 - detector.findAngle(img, 23, 25, 27)
                x, y = pose_lmList[25][1:3]
                cv2.putText(img, str(int(angle)), (x-100,y+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)


            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (10, 80), (0, 100))
            bar = np.interp(angle, (10, 80), (450, 100))
            

            # print(per)

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

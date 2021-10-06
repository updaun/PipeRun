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
        
        seg = img.copy()

        seg = cv2.cvtColor(cv2.flip(seg, 1), cv2.COLOR_BGR2RGB)
        seg.flags.writeable = False

        results = selfie_segmentation.process(seg)

        seg.flags.writeable = True
        seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1

        seg = np.where(condition, seg, bg_image)
        seg = cv2.resize(seg, (640, 480))

        img = cv2.flip(img, 1)
        img = detector.findHolistic(img, draw=False)
        pose_lmList = detector.findPoseLandmark(img, draw=False)
        # print(lmList)
        if len(pose_lmList) != 0:
            if pose_lmList[11][3] > pose_lmList[12][3]:
                angle = 185 - detector.findAngle(seg, 28, 26, 24)
                x, y = pose_lmList[26][1:3] 
                cv2.putText(seg, str(int(angle)), (x-100,y+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            else:
                angle = 185 - detector.findAngle(seg, 23, 25, 27)
                x, y = pose_lmList[25][1:3]
                cv2.putText(seg, str(int(angle)), (x-100,y+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)


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

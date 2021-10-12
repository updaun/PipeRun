from typing import overload
import cv2
import numpy as np
import os
import sys
import time

import modules.HolisticModule as hm
import modules.SegmentationModule as sm

detector = hm.HolisticDetector()
bg_filter = sm.SegmentationFilter()

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


folderPath = "examples\Header_Angle"
myList = os.listdir(folderPath)
# print(myList)

# 덮어씌우는 이미지 리스트
overlayList =[]

# Header 폴더에 image를 상대경로로 지정
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# 비디오 인풋
cap = cv2.VideoCapture(0)


detector = hm.HolisticDetector()
bg_filter = sm.SegmentationFilter()

total_angle = 0

bg_color = (192, 192, 192)
bg_image_path = 'images/_gym.jpg'

# default mode
mode = "exercise"
app_mode = "squat"

header_1 = overlayList[0]
header_2 = overlayList[2]
header_3 = overlayList[4]
header_4 = overlayList[6]
header_5 = overlayList[8]

global count
count = 0
select_count = 0
pTime = 0
dir = 0 

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    seg = img.copy()

    seg = bg_filter.Image(seg, img_path=bg_image_path)
    seg = cv2.resize(seg, (640, 480))

    img = detector.findHolistic(img, draw=False)
    pose_lmList = detector.findPoseLandmark(img, draw=False)

    # 손이 감지가 되었을 때
    if len(pose_lmList) != 0:

        right_hand_x = pose_lmList[16][1]
        left_hand_x = pose_lmList[15][1]
        right_shoulder_x = pose_lmList[12][1]
        left_shoulder_x = pose_lmList[11][1]

        x1, y1 = pose_lmList[20][1:3]
        x2, y2 = pose_lmList[19][1:3]
        foot_x, foot_y = pose_lmList[28][1:3]

        
        
        if (0<x1<100 and y1 < 50) or (0<x2<100 and y2 < 50):
            mode = "select" 
            print("select mode")

        if mode == "exercise":
            if app_mode == "squat":
                print("squat mode activate")
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
                        per = np.interp(total_angle, (50, 300), (0, 100))
                        bar = np.interp(total_angle, (50, 300), (450, 100))
                        
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
                
            elif app_mode == "lunge":
                print("lunge mode activate")
                if pose_lmList[11][3] > pose_lmList[12][3]:
                    rightleg_angle = 185 - detector.findAngle(seg, 28, 26, 24, draw=True)
                    leftleg_angle = 185 - detector.findAngle(seg, 27, 25, 23, draw=True)
                    total_angle = rightleg_angle + leftleg_angle

                    x, y = pose_lmList[26][1:3] 
                    x_core, y_core = pose_lmList[25][1:3]

                    cv2.putText(seg, str(int(rightleg_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    cv2.putText(seg, str(int(leftleg_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    

                    # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                    per = np.interp(total_angle, (80, 175), (0, 100))
                    bar = np.interp(total_angle, (80, 175), (450, 100))
                
                else:
                    rightleg_angle = detector.findAngle(seg, 28, 26, 24, draw=True) - 170
                    leftleg_angle = detector.findAngle(seg, 27, 25, 23, draw=True) - 170
                    total_angle = rightleg_angle + leftleg_angle

                    x, y = pose_lmList[26][1:3] 
                    x_core, y_core = pose_lmList[25][1:3]

                    cv2.putText(seg, str(int(rightleg_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    cv2.putText(seg, str(int(leftleg_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    

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
                
            elif app_mode == "knee up":
                print("knee up mode activate")
                if pose_lmList[11][1] > pose_lmList[12][1]:
                    left_arm_angle = detector.findAngle(seg, 24, 12, 14, draw=True)
                    left_core_angle = 185 - detector.findAngle(seg, 26, 24, 12, draw=True)

                    x, y = pose_lmList[12][1:3] 
                    x_core, y_core = pose_lmList[24][1:3]

                    cv2.putText(seg, str(int(left_arm_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    cv2.putText(seg, str(int(left_core_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                    right_arm_angle = detector.findAngle(seg, 13, 11, 23, draw=True)
                    right_core_angle = 190 - detector.findAngle(seg, 11, 23, 25, draw=True)
                    

                    x, y = pose_lmList[11][1:3]
                    x_core, y_core = pose_lmList[23][1:3]

                    cv2.putText(seg, str(int(right_arm_angle)), (x+40,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    cv2.putText(seg, str(int(right_core_angle)), (x_core+40,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                    if left_arm_angle > 160:
                        left_arm_angle = 160
                    if right_arm_angle > 160:
                        right_arm_angle = 160
                    

                    if pose_lmList[27][2] > pose_lmList[28][2]:
                        total_angle = left_arm_angle + left_core_angle
                    else: 
                        total_angle = right_arm_angle + right_core_angle
                    
                    # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                per = np.interp(total_angle, (150, 200), (0, 100))
                bar = np.interp(total_angle, (150, 200), (450, 100))
                
                
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
                
            elif app_mode == "side lateral raise":
                print("side lateral raise mode activate")
                if pose_lmList[11][1] > pose_lmList[12][1]:
                    left_arm_angle = detector.findAngle(seg, 24, 12, 14, draw=True)
                    # left_core_angle = 185 - detector.findAngle(img, 16, 14, 12, draw=False)

                    x, y = pose_lmList[12][1:3] 
                    # x_core, y_core = pose_lmList[14][1:3]

                    cv2.putText(seg, str(int(left_arm_angle)), (x-60,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    # cv2.putText(img, str(int(left_core_angle)), (x_core-60,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                    right_arm_angle = detector.findAngle(seg, 13, 11, 23, draw=True)
                    # right_core_angle = 190 - detector.findAngle(img, 11, 13, 15, draw=False)
                    

                    x, y = pose_lmList[11][1:3]
                    # x_core, y_core = pose_lmList[13][1:3]

                    cv2.putText(seg, str(int(right_arm_angle)), (x+40,y+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                    # cv2.putText(img, str(int(right_core_angle)), (x_core+40,y_core+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                    total_angle = left_arm_angle + right_arm_angle
                    
                    # cv2.putText(img, str(int(total_angle)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                per = np.interp(total_angle, (40, 150), (0, 100))
                bar = np.interp(total_angle, (40, 150), (450, 100))
                
                
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
                    
            seg[0:50, 0:100] = header_5


        elif mode == "select":
            header_5 = overlayList[9]
        # Checking for the click
            if x1 < 100:
                # walking 
                if 90<=y1<190:
                    print("squat mode")
                    header_1 = overlayList[1]
                    header_2 = overlayList[2]
                    header_3 = overlayList[4]
                    header_4 = overlayList[6]
                    
                    app_mode = "squat"
                    select_count += 1
                    if select_count > 10:
                        select_count = 0
                        app_mode = "squat"
                        mode = "exercise"
                        header_5 = overlayList[8]

                # running
                elif 290<=y1<390:
                    print("lunge mode")
                    header_1 = overlayList[0]
                    header_2 = overlayList[3]
                    header_3 = overlayList[4]
                    header_4 = overlayList[6]

                    app_mode = "lunge"
                    select_count += 1
                    if select_count > 10:
                        select_count = 0
                        app_mode = "lunge"
                        mode = "exercise"
                        header_5 = overlayList[8]

            elif x2 > 540:
                # jumping
                if 90<=y2<190:
                    print("knee up mode")
                    header_1 = overlayList[0]
                    header_2 = overlayList[2]
                    header_3 = overlayList[5]
                    header_4 = overlayList[6]

                    app_mode = "knee up"
                    select_count += 1
                    if select_count > 10:
                        select_count = 0
                        app_mode = "knee up"
                        mode = "exercise"
                        header_5 = overlayList[8]
                    
                # air rope
                elif 290<=y2<390:
                    print("side lateral raise mode")
                    header_1 = overlayList[0]
                    header_2 = overlayList[2]
                    header_3 = overlayList[4]
                    header_4 = overlayList[7]

                    app_mode = "side lateral raise"
                    select_count += 1
                    if select_count > 10:
                        select_count = 0
                        app_mode = "side lateral raise"
                        mode = "exercise"
                        header_5 = overlayList[8]
            
            print(count)

            seg[90:190, 0:100] = header_1
            seg[290:390, 0:100] = header_2
            seg[90:190, 540:640] = header_3
            seg[290:390, 540:640] = header_4
            seg[0:50, 0:100] = header_5
            
            

            

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        #cv2.putText(img, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

        cv2.imshow("Image", seg)    


    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()




from typing import overload
import cv2
import numpy as np
import os
import sys
import time
import pygame

import modules.HolisticModule as hm
import modules.SegmentationModule as sm

detector = hm.HolisticDetector()
bg_filter = sm.SegmentationFilter()

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


folderPath = "examples/Header_Angle"
backgroundfolderPath = "examples/background"
myList = os.listdir(folderPath)
background_myList = os.listdir(backgroundfolderPath)
# print(background_myList)

# 덮어씌우는 이미지 리스트
overlayList =[]
backgroundList = []

# Header 폴더에 image를 상대경로로 지정
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

for imPath in background_myList:
    image_path = f'{backgroundfolderPath}/{imPath}'
    backgroundList.append(image_path)
# print(backgroundList)

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
header = overlayList[10]

total_count = 0

squat_select_count = 0
lunge_select_count = 0
kneeup_select_count = 0
sll_select_count = 0

pTime = 0
dir = 0 

count_text_color = (10,10,10)
count_backgound_color = (245,245,245)

HP = 100
cal = 0
difficulty = 10


sound_control_count = 0

# sound
sounds = {}  # 빈 딕셔너리 생성
pygame.mixer.init()
sounds["pose_ok"] = pygame.mixer.Sound("examples\Assets\Sounds\pose_ok.wav")  # 재생할 파일 설정
sounds["pose_ok"].set_volume(0.5)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["get_score"] = pygame.mixer.Sound("examples\Assets\Sounds\get_score.wav")  # 재생할 파일 설정
sounds["get_score"].set_volume(0.5)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["click"] = pygame.mixer.Sound("examples\Assets\Sounds\click.wav")  # 재생할 파일 설정
sounds["click"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["back"] = pygame.mixer.Sound("examples/Assets/Sounds/back.wav")  # 재생할 파일 설정
sounds["back"].set_volume(0.1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["back"].play()


sounds["g1"] = pygame.mixer.Sound("examples\Assets\Sounds\g1.wav")  # 재생할 파일 설정
sounds["g1"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["g2"] = pygame.mixer.Sound("examples\Assets\Sounds\g2.wav")  # 재생할 파일 설정
sounds["g2"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["g3"] = pygame.mixer.Sound("examples\Assets\Sounds\g3.wav")  # 재생할 파일 설정
sounds["g3"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["g4"] = pygame.mixer.Sound("examples\Assets\Sounds\g4.wav")  # 재생할 파일 설정
sounds["g4"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["g5"] = pygame.mixer.Sound("examples\Assets\Sounds\g5.wav")  # 재생할 파일 설정
sounds["g5"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["g6"] = pygame.mixer.Sound("examples\Assets\Sounds\g6.wav")  # 재생할 파일 설정
sounds["g6"].set_volume(0.5)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    seg = img.copy()

    # 0 15 30 45 60 75 90
    # 6 5  4  3  2  1  0
    if HP == 0:
        bg_image_path = backgroundList[6]
    elif HP == 1:
        sounds["g5"].play()
    elif HP < 15:
        bg_image_path = backgroundList[5]
    elif HP == 15:
        sounds["g4"].play()
    elif HP < 30:
        bg_image_path = backgroundList[4]
    elif HP == 30:
        sounds["g3"].play()
    elif HP < 45:
        bg_image_path = backgroundList[3]
    elif HP == 45:        
        sounds["g2"].play()
    elif HP < 60:
        bg_image_path = backgroundList[2]
    elif HP == 60: 
        sounds["g1"].play()
    elif HP < 75:
        bg_image_path = backgroundList[1]
    elif HP == 75:
        print("test")
        sounds["g6"].play()
    else:
        bg_image_path = backgroundList[0]


    if bg_image_path != None:
        seg = bg_filter.Image(seg, img_path=bg_image_path)
    seg = cv2.resize(seg, (640, 480))

    img = detector.findHolistic(img, draw=False)
    pose_lmList = detector.findPoseLandmark(img, draw=False)

    # 신체 감지가 되었을 때
    if len(pose_lmList) != 0:

        right_hand_x = pose_lmList[16][1]
        left_hand_x = pose_lmList[15][1]
        right_shoulder_x = pose_lmList[12][1]
        left_shoulder_x = pose_lmList[11][1]

        x1, y1 = pose_lmList[20][1:3]
        x2, y2 = pose_lmList[19][1:3]
        foot_x, foot_y = pose_lmList[28][1:3]

        
        
        if (540<x1<=640 and y1 < 50) or (540<x2<=640 and y2 < 50):
            mode = "select" 
            # sounds["click"].play()
            print("select mode")

        if mode == "exercise":
            HP -= 0.5
            if HP < 0:
                HP = 0 
                wording = "Game Over"
                coords = (125, 330)
                cv2.rectangle(seg,(coords[0], coords[1]-50), (coords[0]+len(wording)*40, coords[1]+20), (230, 230, 230), -1) 
                cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 0, 200), 3, cv2.LINE_AA)

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
                color = (0,0,200)
                if per == 100:
                    color = (0,200,0)        
                    if dir == 0:
                        sounds["pose_ok"].play()
                        total_count += 0.5
                        dir = 1
                        HP += difficulty
                if per == 0:
                    color = (0,200,0)        
                    if dir == 1:
                        sounds["get_score"].play()
                        total_count += 0.5
                        dir = 0
                        cal += 55
                        HP += difficulty

                # Draw bar
                cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                cv2.putText(seg, f'{int(per)}%', (565, 80),
                            cv2.LINE_AA, 0.8, color, 2)        
        
                # Display Class
                cv2.rectangle(seg, (0,0), (170, 60), (16, 117, 245), -1)
                cv2.putText(seg, 'COUNT'
                            , (90,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(int(total_count))
                            , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(seg, 'HP'
                            , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(HP)
                            , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
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
                color = (0,0,200)
                if per == 100:
                    color = (0,200,0)        
                    if dir == 0:
                        sounds["pose_ok"].play()
                        total_count += 0.5
                        dir = 1
                        HP += difficulty
                if per == 0:
                    color = (0,200,0)        
                    if dir == 1:
                        sounds["get_score"].play()
                        total_count += 0.5
                        dir = 0
                        cal += 33
                        HP += difficulty

                # Draw bar
                cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                cv2.putText(seg, f'{int(per)}%', (565, 80),
                            cv2.LINE_AA, 0.8, color, 2)        
        
                # Display Class
                cv2.rectangle(seg, (0,0), (170, 60), (16, 117, 245), -1)
                cv2.putText(seg, 'COUNT'
                            , (90,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(int(total_count))
                            , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(seg, 'HP'
                            , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(HP)
                            , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
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
                color = (0,0,200)
                if per == 100:
                    color = (0,200,0)        
                    if dir == 0:
                        sounds["pose_ok"].play()
                        total_count += 0.5
                        dir = 1
                        HP += difficulty
                if per == 0:
                    color = (0,200,0)        
                    if dir == 1:
                        sounds["get_score"].play()
                        total_count += 0.5
                        dir = 0
                        cal += 33
                        HP += difficulty

                # Draw bar
                cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                cv2.putText(seg, f'{int(per)}%', (565, 80),
                            cv2.LINE_AA, 0.8, color, 2)        
        
                # Display Class
                cv2.rectangle(seg, (0,0), (170, 60), (16, 117, 245), -1)
                cv2.putText(seg, 'COUNT'
                            , (90,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(int(total_count))
                            , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(seg, 'HP'
                            , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(HP)
                            , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
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
                color = (0,0,200)
                if per == 100:
                    color = (0,200,0)        
                    if dir == 0:
                        sounds["pose_ok"].play()
                        total_count += 0.5
                        dir = 1
                        HP += difficulty
                if per == 0:
                    color = (0,200,0)        
                    if dir == 1:
                        sounds["get_score"].play()
                        total_count += 0.5
                        dir = 0
                        cal += 33
                        HP += difficulty

                # Draw bar
                cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                cv2.putText(seg, f'{int(per)}%', (565, 80),
                            cv2.LINE_AA, 0.8, color, 2)        
        
                # Display Class
                cv2.rectangle(seg, (0,0), (170, 60), (16, 117, 245), -1)
                cv2.putText(seg, 'COUNT'
                            , (90,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(int(total_count))
                            , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(seg, 'HP'
                            , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(seg, str(HP)
                            , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
            seg[0:50, 540:640] = header_5


        elif mode == "select":

            wording = "Total Calories : "
            coords = (130, 120)
            cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*20, coords[1]-30), (230, 230, 230), -1) 
            cv2.putText(seg, wording + str(cal), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)

            header_5 = overlayList[9]
            
            # header = cv2.imread('examples\Header\mute.png/mute.png')
            
        # Checking for the click
            if x1 < 100:
                if y1<50:
                    sounds["back"].stop()
                    break
                # walking 
                if 90<=y1<190:
                    print("squat mode")
                    header_1 = overlayList[1]
                    header_2 = overlayList[2]
                    header_3 = overlayList[4]
                    header_4 = overlayList[6]
                    
                    app_mode = "squat"
                    squat_select_count += 1
                    lunge_select_count = 0
                    kneeup_select_count = 0
                    sll_select_count = 0
                    if squat_select_count > 10:
                        squat_select_count = 0
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
                    lunge_select_count += 1
                    squat_select_count = 0
                    kneeup_select_count = 0
                    sll_select_count = 0
                    if lunge_select_count > 10:
                        lunge_select_count = 0
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
                    kneeup_select_count += 1
                    lunge_select_count = 0
                    squat_select_count = 0
                    sll_select_count = 0
                    if kneeup_select_count > 10:
                        kneeup_select_count = 0
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
                    sll_select_count += 1
                    kneeup_select_count = 0
                    lunge_select_count = 0
                    squat_select_count = 0
                    if sll_select_count > 10:
                        sll_select_count = 0
                        app_mode = "side lateral raise"
                        mode = "exercise"
                        header_5 = overlayList[8]
            
            print(total_count)

            seg[90:190, 0:100] = header_1
            seg[290:390, 0:100] = header_2
            seg[90:190, 540:640] = header_3
            seg[290:390, 540:640] = header_4
            seg[0:50, 540:640] = header_5
            seg[0:50, 0:100] = header

            
            
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if HP > 100:
            HP = 100
        #cv2.putText(img, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

        cv2.imshow("Image", seg)    

    if cv2.waitKey(1) & 0xFF == 27:
        break

    else:
        wording = "Please Appear On The Screen"
        coords = (80, 250)
        cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*18, coords[1]-30), (230, 230, 230), -1) 
        cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)


        cv2.imshow("Image", seg)  

cv2.destroyAllWindows()




import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import math

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import modules.HolisticModule as hm

mpDraw = mp.solutions.drawing_utils #mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands           #mpHands = mp.solutions.hands
hands = mpHands.Hands()                 #hands = mpHands.Hands()

# class 객체 생성
detector = hm.HolisticDetector()

score = 0

x_enemy = random.randint(50, 600)
y_enemy = random.randint(50, 400)


def enemy():
    global score, x_enemy, y_enemy
    # x_enemy=random.randint(50,600)
    # y_enemy=random.randint(50,400)
    cv2.circle(image, (x_enemy, y_enemy), 25, (0, 200, 0), 5)
    # score=score+1



video = cv2.VideoCapture(0)


while True:
    _, frame = video.read()
    image = frame.copy()
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # image = cv2.flip(image, 1)

    # imageHeight, imageWidth, _ = frame.shape

    # results = hands.process(image)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Holistic process ( 위에 주석한 4줄과 같은 효과)
    image = detector.findHolistic(image, draw=True)

    # 왼손 좌표 리스트
    LefthandLandmarkList = detector.findLefthandLandmark(image)
    # 오른손 좌표 리스트
    RighthandLandmarkList = detector.findRighthandLandmark(image)

    enemy()

    if len(LefthandLandmarkList) != 0:
        cv2.circle(image, (LefthandLandmarkList[8][1:3]), 25, (0, 200, 0), 5)
        if abs(LefthandLandmarkList[8][1]-x_enemy) < 10 and abs(LefthandLandmarkList[8][2]-y_enemy) < 10:
            print("found")
            x_enemy = random.randint(50, 600)
            y_enemy = random.randint(50, 400)
            score = score + 1
            enemy()
    if len(RighthandLandmarkList) != 0:
        cv2.circle(image, (RighthandLandmarkList[8][1:3]), 25, (0, 200, 0), 5)
        if abs(RighthandLandmarkList[8][1]-x_enemy) < 10 and abs(RighthandLandmarkList[8][2]-y_enemy) < 10:
            print("found")
            x_enemy = random.randint(50, 600)
            y_enemy = random.randint(50, 400)
            score = score + 1
            enemy()
    
    image = cv2.flip(image, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 255)
    text = cv2.putText(image, "Score", (480, 30), font, 1, color, 4, cv2.LINE_AA)
    text = cv2.putText(image, str(score), (590, 30), font, 1, color, 4, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', image)
    # time.sleep(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print(score)
        break

video.release()
cv2.destroyAllWindows()
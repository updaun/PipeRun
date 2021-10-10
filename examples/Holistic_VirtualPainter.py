from typing import overload
import cv2
import numpy as np
import os
import sys

import modules.HolisticModule as hm
import modules.HandTrackingModule as htm
# from datetime import datetime
# import pytz

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

###################################
brushThickness = 15
eraserThickness = 50
###################################

folderPath = "examples\Header"
myList = os.listdir(folderPath)
# print(myList)

# 덮어씌우는 이미지 리스트
overlayList =[]

# 사진을 찍으면 카운트가 증가합니다.
img_counter = 1

# time_zone = pytz.timezone('Asia/Seoul')

# now = datetime.now(time_zone)

# current_time = now.strftime("%H%M")

# Header 폴더에 image를 상대경로로 지정
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

# default color
drawColor = (230, 230, 230)

# 비디오 인풋
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

# detectionCon를 지정하여 손을 정확히 찾는다.(그림을 잘 그리기 위해서 수정)
# detector = htm.handDetector(detectionCon=0.85)
detector = hm.HolisticDetector()

# previous 좌표
xp, yp = 0, 0

# 검정색 캔버스
imgCanvas = np.zeros((480, 848, 3), np.uint8)
# 하얀색 캔버스
whiteCanvas = imgCanvas + 255
# 하얀색 캔버스
imgInv = np.zeros((480, 848, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    # print(img.shape)


    # 사용자 인식을 위해 Filp
    img = cv2.flip(img, 1)


    # 2. Find Hand Landmarks
    # detector 라는 클래스를 선언하고 손을 찾는다.
    img = detector.findHolistic(img)

    # 손의 좌표를 lmList에 저장한다.( 손은 21개의 좌표를 포함 )
    lmList = detector.findRighthandLandmark(img)

    

    # 손이 감지가 되었을 때
    if len(lmList) != 0:
        # print(lmList)

        # tip of Index fingers(검지 손가락의 좌표)
        x1, y1 = lmList[8][1:3]
        # tip of Middle fingers(중지 손가락의 좌표)
        x2, y2 = lmList[12][1:3]

        # 3. Check which fingers are up
        # 손가락이 펴져있는지 아닌지 체크하는 함수
        fingers = detector.right_hand_fingersUp()
        print(fingers)

        # 4. If Selection Mode - Two fingers are up
        # 검지와 중지가 펴져있을 때
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # Checking for the click
            if y1 < 100:
                # Dark Gray -> while
                # 다크 그레이색이 선택된 Header의 이미지가 오버레이 된다.
                if 200<x1<250:
                    header = overlayList[0]
                    drawColor = (230, 230, 230)
                # Deep Blue
                elif 325<x1<375:
                    header = overlayList[1]
                    drawColor = (122, 6, 6)
                # Deep Green
                elif 400<x1<450:
                    header = overlayList[2]
                    drawColor = (8, 102, 5)
                # Orange
                elif 500<x1<550:
                    header = overlayList[3]
                    drawColor = (0, 69, 255)
                # Yellow
                elif 625<x1<675:
                    header = overlayList[4]
                    drawColor = (0, 255, 255)
                # Eraser 
                elif 725<x1<840:
                    header = overlayList[5]
                    drawColor = (255, 255, 255)
                
            # 제대로 선택되었는지 확인하기 위해 손에도 색을 표시한다.(사각형)
            cv2.rectangle(img, (x1, y1-20), (x2, y2+20), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        # 검지 손가락은 펴지고, 중지손가락은 펴지지 않을 상태
        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # drawColor 흰색이면 -> 지우개가 선택된 상태
            if drawColor == (255,255,255):
                cv2.line(img, (xp, yp), (x1,y1), (0,0,0), eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), (0,0,0), eraserThickness)
                cv2.line(whiteCanvas, (xp, yp), (x1,y1), (255,255,255), eraserThickness)
                cv2.circle(img, (x1, y1), int(eraserThickness/2)+2, (230,230,230), cv2.FILLED)
            # drawColor 지정된 색으로 그림을 그린다.(지우개가 아닌 상태)
            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                cv2.line(img, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(whiteCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)

            xp, yp = x1, y1
    
    imgCanvas = cv2.flip(imgCanvas, 1)

    # 각 캔버스를 더하고 곱하는 부분(마스크 처리)
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # Setting the header image
    # 헤더를 오버레이하는 부분
    img[0:100, 0:848] = header
    # blend img
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("whiteCanvas", whiteCanvas)
    # cv2.imshow("ImgInv", imgInv)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', whiteCanvas)
        print("Save Canvas Successfully")
        break

    # Spacebar 또는 Return 누르면 whiteCanvas 저장
    if cv2.waitKey(1) & 0xFF == 32 or cv2.waitKey(1) & 0xFF == 13:
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', whiteCanvas)
        print("Save Canvas Successfully")
        img_counter += 1

cv2.destroyAllWindows()




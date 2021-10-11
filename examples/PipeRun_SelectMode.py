from typing import overload
import cv2
import numpy as np
import os
import sys

import modules.HolisticModule as hm

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

header_1 = overlayList[0]
header_2 = overlayList[2]
header_3 = overlayList[4]
header_4 = overlayList[6]

# default color
drawColor = (230, 230, 230)

# 비디오 인풋
cap = cv2.VideoCapture(0)

detector = hm.HolisticDetector()

# previous 좌표
xp, yp = 0, 0

while True:

    # 1. Import image
    success, img = cap.read()
    print(img.shape)


    # 사용자 인식을 위해 Filp
    img = cv2.flip(img, 1)


    # 2. Find Hand Landmarks
    # detector 라는 클래스를 선언하고 손을 찾는다.
    img = detector.findHolistic(img)

    # 손의 좌표를 lmList에 저장한다.( 손은 21개의 좌표를 포함 )
    lmList = detector.findPoseLandmark(img)

    

    # 손이 감지가 되었을 때
    if len(lmList) != 0:
        # print(lmList)

        # tip of Index fingers(검지 손가락의 좌표)
        x1, y1 = lmList[20][1:3]
        x2, y2 = lmList[19][1:3]
        
        # 4. If Selection Mode - Two fingers are up
        # 검지와 중지가 펴져있을 때
        # print("Selection Mode")
        # Checking for the click
        if x1 < 100:
            # walking 
            if 90<=y1<190:
                header_1 = overlayList[1]
                header_2 = overlayList[2]
                header_3 = overlayList[4]
                header_4 = overlayList[6]
                drawColor = (230, 230, 230)
            # running
            elif 290<=y1<390:
                header_1 = overlayList[0]
                header_2 = overlayList[3]
                header_3 = overlayList[4]
                header_4 = overlayList[6]
                drawColor = (122, 6, 6)
        elif x2 > 540:
            # jumping
            if 90<=y2<190:
                header_1 = overlayList[0]
                header_2 = overlayList[2]
                header_3 = overlayList[5]
                header_4 = overlayList[6]
                drawColor = (8, 102, 5)
            # air rope
            elif 290<=y2<390:
                header_1 = overlayList[0]
                header_2 = overlayList[2]
                header_3 = overlayList[4]
                header_4 = overlayList[7]
                drawColor = (0, 69, 255)
            
        # 제대로 선택되었는지 확인하기 위해 손에도 색을 표시한다.(사각형)
        cv2.rectangle(img, (x1, y1-20), (x1+20, y1), drawColor, cv2.FILLED)


        # drawColor 흰색이면 -> 지우개가 선택된 상태
        if drawColor == (255,255,255):
            pass

        else:
            pass


    img[90:190, 0:100] = header_1
    img[290:390, 0:100] = header_2
    img[90:190, 540:640] = header_3
    img[290:390, 540:640] = header_4

    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()




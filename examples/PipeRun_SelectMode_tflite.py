from typing import overload
import cv2
import numpy as np
import os
import sys

import modules.HolisticModule as hm

import tensorflow as tf

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



# Load TFLite model and allocate tensors.
interpreter_1 = tf.lite.Interpreter(model_path="models/walking_model.tflite")
interpreter_1.allocate_tensors()
interpreter_2 = tf.lite.Interpreter(model_path="models/running_model.tflite")
interpreter_2.allocate_tensors()
interpreter_3 = tf.lite.Interpreter(model_path="models/jumping_model.tflite")
interpreter_3.allocate_tensors()
interpreter_4 = tf.lite.Interpreter(model_path="models/airrope_model.tflite")
interpreter_4.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_1.get_input_details()
output_details = interpreter_1.get_output_details()

folderPath = "examples\Header"
myList = os.listdir(folderPath)
# print(myList)

# 덮어씌우는 이미지 리스트
overlayList =[]

# Header 폴더에 image를 상대경로로 지정
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header_1 = overlayList[1]
header_2 = overlayList[2]
header_3 = overlayList[4]
header_4 = overlayList[6]

# default color
drawColor = (230, 230, 230)

# 비디오 인풋
cap = cv2.VideoCapture(0)


detector = hm.HolisticDetector()

seq = []
action_seq = []
last_action = None

actions = ['stop', 'fit']
seq_length = 30

# default mode
app_mode = "walking"

while True:

    # 1. Import image
    success, img = cap.read()

    # 사용자 인식을 위해 Filp
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    # detector 라는 클래스를 선언하고 손을 찾는다.
    result = detector.findHolisticwithResults(img)
    # 손의 좌표를 lmList에 저장한다.( 손은 21개의 좌표를 포함 )
    lmList = detector.findPoseLandmark(img)

    # 손이 감지가 되었을 때
    if len(lmList) != 0:
        # print(lmList)

        # tip of Index fingers(검지 손가락의 좌표)
        x1, y1 = lmList[20][1:3]
        x2, y2 = lmList[19][1:3]
        
        joint = np.zeros((33, 4))
        for j, lm in enumerate(result.pose_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # Compute angles between joints
        v1 = joint[[0,0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,13,14,15,15,15,16,16,16,17,18,19,20,21,22,23,24,25,26,27,27,28,28,29,30,31,32], :3] # Parent joint
        v2 = joint[[1,4,2,3,7,5,6,8,3,6,0, 0,13,23,14,24,15,16,21,19,17,22,20,18,19,20,15,16,15,16,25,26,27,28,31,29,32,30,31,32,27,28], :3] # Child joint
        
        v = v2 - v1 # [20, 3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],:], 
                    v[[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],:])) 

        angle = np.degrees(angle) / 360 # Convert radian to degree

        d = np.concatenate([joint.flatten(), angle])

        seq.append(d)

        if len(seq) < seq_length:
            continue

        # 시퀀스 데이터와 넘파이화
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)


        # Checking for the click
        if x1 < 100:
            # walking 
            if 90<=y1<190:
                print("walking mode")
                header_1 = overlayList[1]
                header_2 = overlayList[2]
                header_3 = overlayList[4]
                header_4 = overlayList[6]

                app_mode = "walking"

            # running
            elif 290<=y1<390:
                print("running mode")
                header_1 = overlayList[0]
                header_2 = overlayList[3]
                header_3 = overlayList[4]
                header_4 = overlayList[6]

                app_mode = "running"

        elif x2 > 540:
            # jumping
            if 90<=y2<190:
                print("jumping mode")
                header_1 = overlayList[0]
                header_2 = overlayList[2]
                header_3 = overlayList[5]
                header_4 = overlayList[6]

                app_mode = "jumping"
                
            # air rope
            elif 290<=y2<390:
                print("air rope mode")
                header_1 = overlayList[0]
                header_2 = overlayList[2]
                header_3 = overlayList[4]
                header_4 = overlayList[7]

                app_mode = "air rope"
                
        if app_mode == "walking":
            print("walking mode activate")
            interpreter_1.set_tensor(input_details[0]['index'], input_data)
            interpreter_1.invoke()
            y_pred = interpreter_1.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
        elif app_mode == "running":
            print("running mode activate")
            interpreter_2.set_tensor(input_details[0]['index'], input_data)
            interpreter_2.invoke()
            y_pred = interpreter_2.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
        elif app_mode == "jumping":
            print("jumping mode activate")
            interpreter_3.set_tensor(input_details[0]['index'], input_data)
            interpreter_3.invoke()
            y_pred = interpreter_3.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
        elif app_mode == "air rope":
            print("air rope mode activate")
            interpreter_4.set_tensor(input_details[0]['index'], input_data)
            interpreter_4.invoke()
            y_pred = interpreter_4.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))
            
        
        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action
        
        # cv2.putText(img, f'{this_action.upper()}', org=(int(result.face_landmarks.landmark[0].x * img.shape[1]), int(result.face_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # 도식의 기준 좌표 생성 (왼쪽 귀)
        coords = tuple(np.multiply(
                        np.array(
                            (result.pose_landmarks.landmark[7].x+20, 
                                result.pose_landmarks.landmark[7].y))
                    , [640,480]).astype(int))
        
        # 사각형 그리기
        cv2.rectangle(img, 
                        # 사각형의 왼쪽 위
                        (coords[0], coords[1]+5), 
                        # 사각형의 오른쪽 아래
                        (coords[0]+len(this_action)*20, coords[1]-30), 
                        (245, 117, 16), -1) # -1 사각형 안을 가득 채운다.
        # 어떤 액션인지 글자 표시
        cv2.putText(img, this_action, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Get status box
        cv2.rectangle(img, (0,0), (170, 60), (245, 117, 16), -1)
        
        # Display Class
        cv2.putText(img, 'CLASS'
                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, this_action.split(' ')[0]
                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display Probability
        cv2.putText(img, 'PROB'
                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(round(y_pred[0][np.argmax(y_pred[0])],2))
                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    img[90:190, 0:100] = header_1
    img[290:390, 0:100] = header_2
    img[90:190, 540:640] = header_3
    img[290:390, 540:640] = header_4

    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()




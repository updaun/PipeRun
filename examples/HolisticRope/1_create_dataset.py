import cv2
import mediapipe as mp
import numpy as np
import time, os

# 기쁨 -> easy  나쁨 -> difficult
# 학습자의 이해도 분석
actions = ['rope', 'stop']
seq_length = 30
# 30초 동안 촬영해서 데이터를 확보
secs_for_action = 30

# Holistic 사용
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Holistic Class 생성 (감지기 역할 수행)
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠으로 이미지를 입력
cap = cv2.VideoCapture(0)

# FPS 계산을 위한 현재 시간 계산
created_time = int(time.time())

# 데이터 저장 폴더 생성
os.makedirs('dataset', exist_ok=True)

# 캠이 정상적으로 작동한다면
while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        # img = cv2.flip(img, 1)
        # 데이터 확보 시작 전 3초간 알림
        # 화면에 글자를 새기는 구문 ( 이미지, 쓸 내용, 시작 좌표, 폰트종류, 폰트스케일, 글자색, 글자 두께 )
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # 우리에게 창을 생성해서 보여주는 부분
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        # 데이터 촬영 시작 시간
        start_time = time.time()

        # 30초 동안만 반복문 구동
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read() # 원본 사진

            # img = cv2.flip(img, 1)
            # Mediapipe를 구동시키기 위해 RGB로 변경
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Mediapipe 통과
            result = holistic.process(img)

            # 다시 Numpy 연산을 위해 BGR로 변경
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # face = result.face_landmarks.landmark
            # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # 얼굴 좌표가 있을 때만 구동
            if result.pose_landmarks is not None:
                # for res in result.face_landmarks:
                # 빈 Numpy array 생성( 얼굴 좌표 개수 만큼 생성 )
                joint = np.zeros((33, 4))
                for j, lm in enumerate(result.pose_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,13,14,15,15,15,16,16,16,17,18,19,20,21,22,23,24,25,26,27,27,28,28,29,30,31,32], :3] # Parent joint
                v2 = joint[[1,4,2,3,7,5,6,8,3,6,0, 0,13,23,14,24,15,16,21,19,17,22,20,18,19,20,15,16,15,16,25,26,27,28,31,29,32,30,31,32,27,28], :3] # Child joint

                v = v2 - v1 # [20, 3]
                # Normalize v
                # 0 ~ 1사이의 값으로 변환
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                # 각도를 구하는 부분
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],:], 
                    v[[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],:])) 

                # 라디안값을 도로 변환
                angle = np.degrees(angle) / 360 # Convert radian to degree

                # 빈 넘파이 어레이 생성 
                angle_label = np.array([angle], dtype=np.float32)

                # 생성한 어레이에 각도값을 추가한다. 0과 1이 들어갈 예정 (데이터 라벨 작업)
                angle_label = np.append(angle_label, idx)

                # 좌표값 데이터에 각도값 데이터를 연결한다.
                d = np.concatenate([joint.flatten(), angle_label])

                # 최종 데이터에 합산
                data.append(d)

                # 촬영 중 얼굴이 인식되었는지 확인용
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # 얼굴 그림이 그려진 사진을 사용자에게 보여준다.
            cv2.imshow('img', img)

            # ESC 키를 누르면
            if cv2.waitKey(5) & 0xFF == 27:
                # 종료
                break

        # 확보덴 데이터를 np.array화 한다.
        data = np.array(data)
        # print(action, data.shape)
        # 데이터 저장 npy 형식의 데이터로 저장
        # np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

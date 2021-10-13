import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import math
import pygame
import threading
from playsound import playsound
import pandas as pd
import numpy as np
import time
import moviepy.editor as mv
import parselmouth
import pandas as pd
import os.path
import time
from pytube import YouTube
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import modules.PoseModule as pm
import os

# 유튜브 링크 입력으로 다운받아오는 부분
download_path = './demo'

yt_url = input()
if not os.path.exists(download_path):
    os.makedirs(download_path)

video = YouTube(yt_url)
video_type = video.streams.filter(progressive = True, file_extension = "mp4").get_highest_resolution()
video_type.download('./demo/')

# 다운받은 비디오의 이름을 video.mp4로 바꾸기 -> 이러면 노래 하나만 있어야한다는 문제점.
file_list = os.listdir(download_path)
old_name = os.path.join(download_path, file_list[0])
new_name = os.path.join(download_path, "video.mp4")

os.rename(old_name, new_name)

# mp4를 mp3 로 전환
clip = mv.VideoFileClip("demo/video.mp4")
clip.audio.write_audiofile("demo/audio.mp3")

audio_file_path = "demo/audio.mp3"
#ㅈ
Sound = parselmouth.Sound(audio_file_path)

# 1초단위로
formant = Sound.to_formant_burg(time_step=1)
# Pitch값 추출
pitch = Sound.to_pitch()
df = pd.DataFrame({"times": formant.ts()})

times =[]
times.append(formant.ts())

df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))

df.to_csv("demo/pitchdata.csv")



sound = pd.read_csv("pitchdata.csv", index_col=0)
#
df = sound['F0(pitch)']
sound['times'] = sound['times'].astype(int)
#print(sound.isnull().sum())
normal_df =(df-df.min())/(df.max()-df.min())
pitch = normal_df.fillna(2)
sound['F0(pitch)'] = pitch
# pitch_val = pitch.values.tolist()
# if sound['F0(pitch)'] <= 0.3:
#     pass
# elif sound['F0(pitch)'] <= 0.7:
#     pass
# elif sound['F0(pitch)'] <= 1.0:
#     pass

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# 노래 삽입
sounds = {}  # 빈 딕셔너리 생성
pygame.mixer.init()
sounds["slap"] = pygame.mixer.Sound("examples\Assets\Sounds\slap.wav")  # 재생할 파일 설정
sounds["slap"].set_volume(1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["screaming"] = pygame.mixer.Sound("examples\Assets\Sounds\Effect.wav")  # 재생할 파일 설정
sounds["screaming"].set_volume(1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
sounds["song"] = pygame.mixer.Sound("demo/audio.mp3")  # 재생할 파일 설정
sounds["song"].set_volume(1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨

# class 객체 생성
movedetector = pm.poseDetector()
score = 0
note_count = 20
x_enemy = random.randint(50, 600)
y_enemy = random.randint(50, 400)

mp3_time = 0

sounds["song"].play()

# print((sound.iloc[1]['F0(pitch)']))

#     pass
def make_pitch(df, i):
    # i_pitch = df(df['times'] == i)['F0(pitch)']
    i_pitch = df.iloc[i]['F0(pitch)']
    if i_pitch < 0.3:
       result = random.randint(310, 400)
    elif i_pitch >=0.3 and i_pitch<0.6:
       result = random.randint(220, 310)
    elif i_pitch >=0.6 and i_pitch<=1:
       result = random.randint(50, 220)
    elif i_pitch == 2:
       result = random.randint(50, 400)
    print(i_pitch)
    return result


def enemy():
    global score, x_enemy, y_enemy, count
    # pTime = 0
    #
    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime

    # x_enemy=random.randint(50,600)
    # y_enemy=random.randint(50,400)
    cv2.circle(image, (x_enemy, y_enemy), 25, (random.randint(0, 256), random.randint(0, 256),
                                               random.randint(0, 256)), 5)
    # score=score+1


def enemys():
    global score, x_enemy, y_enemy
    # x_enemy=random.randint(50,600)
    # y_enemy=random.randint(50,400)
    cv2.rectangle(image, (x_enemy, y_enemy), (x_enemy + 40, y_enemy + 40),
                  (random.randint(0, 256), random.randint(0, 256),
                   random.randint(0, 256)), 5)


def enemyt():
    global score, x_enemy, y_enemy
    # x_enemy=random.randint(50,600)
    # y_enemy=random.randint(50,400)
    pts = np.array([[x_enemy, y_enemy], [x_enemy + 60, y_enemy], [x_enemy + 30, y_enemy - 49]], np.int32)
    cv2.polylines(image, [pts], True, (random.randint(0, 256), random.randint(0, 256),
                                       random.randint(0, 256)), 3)


video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    image = frame.copy()
    movedetector.findPose(image)
    footlmList = movedetector.findPosition(image, draw=False)

    if note_count == 10:
        mp3_time +=1

    # 원형 노트 생성
    if note_count > 0 and y_enemy < 220:
        note_count -= 1
        cv2.circle(image, (x_enemy, y_enemy), 25,
                   (0, 0, random.randint(60,255)), 5)
    # 삼각형 노트 생성
    elif note_count > 0 and y_enemy >= 220 and y_enemy < 300:
        note_count -= 1
        pts = np.array([[x_enemy, y_enemy], [x_enemy + 60, y_enemy], [x_enemy + 30, y_enemy - 49]], np.int32)
        cv2.polylines(image, [pts], True, (0, 0, random.randint(60,255)), 3)
    # 사각형 노트 생성
    elif note_count > 0 and y_enemy >= 300:
        note_count -= 1
        cv2.rectangle(image, (x_enemy, y_enemy), (x_enemy + 40, y_enemy + 40),
                      (0, 0, random.randint(60,255)), 5)

    # 사용자가 감지를 못 했을 때
    if note_count == 0:
        x_enemy = random.randint(50, 600)
        y_enemy = make_pitch(sound, mp3_time)
        note_count = 25  # random.randint(30,40)


    if len(footlmList) != 0:

        # 내 몸에 손목 그리기
        cv2.circle(image, (footlmList[16][1:3]), 25, (0, 200, 0), 5)
        cv2.circle(image, (footlmList[15][1:3]), 25, (0, 200, 0), 5)

        # 내 몸에 삼각형 그리기
        pts = np.array([[footlmList[25][1], footlmList[25][2]], [footlmList[25][1] + 60, footlmList[25][2]],
                        [footlmList[25][1] + 30, footlmList[25][2] - 49]], np.int32)
        cv2.polylines(image, [pts], True, (0, 200, 0), 3)

        pts = np.array([[footlmList[26][1], footlmList[26][2]], [footlmList[26][1] + 60, footlmList[26][2]],
                        [footlmList[26][1] + 30, footlmList[26][2] - 49]], np.int32)
        cv2.polylines(image, [pts], True, (0, 200, 0), 3)

        # 냬 몸에 사각형 그리기
        cv2.rectangle(image, (footlmList[31][1], footlmList[31][2]),
                      (footlmList[31][1] + 40, footlmList[31][2] + 40),
                      (0, 200, 0), 5)

        cv2.rectangle(image, (footlmList[32][1], footlmList[32][2]),
                      (footlmList[32][1] + 40, footlmList[32][2] + 40),
                      (0, 200, 0), 5)


        if y_enemy < 220:
            # 오른쪽 무릎으로 감지 했을 때
            if abs(footlmList[25][1] - x_enemy) < 30 and abs(footlmList[25][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 왼쪽 무릎으로 감지 했을 때
            elif abs(footlmList[26][1] - x_enemy) < 30 and abs(footlmList[26][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 발로 감지 했을 때
            elif abs(footlmList[31][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 발로 감지 했을 때
            elif abs(footlmList[32][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 손으로 감지 했을 때
            elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
            # 손으로 감지 했을 때
            elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
    # 삼각형일 때
        elif y_enemy >= 220 and y_enemy < 310:
            # 오른쪽 무릎으로 감지 했을 때
            if abs(footlmList[25][1] - x_enemy) < 30 and abs(footlmList[25][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
            # 왼쪽 무릎으로 감지 했을 때
            elif abs(footlmList[26][1] - x_enemy) < 30 and abs(footlmList[26][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
            # 발로 감지 했을 때
            elif abs(footlmList[31][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 발로 감지 했을 때
            elif abs(footlmList[32][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 손으로 감지 했을 때
            elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 손으로 감지 했을 때
            elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
        # 사각형일 때
        elif y_enemy>=310:
            # 오른쪽 무릎으로 감지 했을 때
            if abs(footlmList[25][1] - x_enemy) < 30 and abs(footlmList[25][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 왼쪽 무릎으로 감지 했을 때
            elif abs(footlmList[26][1] - x_enemy) < 30 and abs(footlmList[26][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 발로 감지 했을 때
            elif abs(footlmList[31][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
            # 발로 감지 했을 때
            elif abs(footlmList[32][1] - x_enemy) < 30 and abs(footlmList[32][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("found")
                sounds["slap"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score + 1
            # 손으로 감지 했을 때
            elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1
            # 손으로 감지 했을 때
            elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                x_enemy, y_enemy = 1000, 1000
                print("fail")
                sounds["screaming"].play()  # 음원 재생
                note_count = 10
                mp3_time += 1
                score = score - 1



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

    # if cv2.waitKey(1) & 0xFF == 27:
    #     sounds["slap"].play()
    # print(score)
    # break

video.release()
cv2.destroyAllWindows()
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

audio_path = "demo/audio.mp3"
if os.path.exists(audio_path):
    os.remove("demo/audio.mp3")

# mp4를 mp3 로 전환
clip = mv.VideoFileClip("demo/"+video.title+".mp4")
clip.audio.write_audiofile(audio_path)
Sound = parselmouth.Sound(audio_path)

# 1초단위로
formant = Sound.to_formant_burg(time_step=1)
# Pitch값 추출
pitch = Sound.to_pitch()
df = pd.DataFrame({"times": formant.ts()})

times =[]
times.append(formant.ts())

df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))

df.to_csv("demo/pitchdata.csv")
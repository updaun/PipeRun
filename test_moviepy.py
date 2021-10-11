# 모듈 로딩 후 오디오 추출
import moviepy.editor as mv
import parselmouth
import pandas as pd
import os.path
import time
# import pytube



clip = mv.VideoFileClip("demo/running.mp4")
clip.audio.write_audiofile("demo/extracted_audio.mp3")

audio_file_path = "demo/extracted_audio.mp3"
# path = os.getcwd()
# file_list = os.listdir(path)  # 경로 내 파일들
# mp3_list = [file for file in file_list if file.endswith('.mp3')]  # 중에서 mp3 파일만

# test = os.path.join(path, mp3_list[0]) # 노래 한곡만 있다고 생각하고 0번

Sound = parselmouth.Sound(audio_file_path)

# 1초단위로
formant = Sound.to_formant_burg(time_step=1)
# Pitch값 추출
pitch = Sound.to_pitch()
df = pd.DataFrame({"times": formant.ts()})

times =[]
times.append(formant.ts())
print(times)

df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))

print(df)

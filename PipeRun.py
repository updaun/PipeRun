from re import T
from mediapipe.python.solutions.face_mesh import FACE_CONNECTIONS
from numpy.core.fromnumeric import size
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import tensorflow as tf

import os
import sys
import pygame

from typing import overload

import examples.modules.HolisticModule as hm
import examples.modules.SegmentationModule as sm

from mediapipe.framework.formats import landmark_pb2
import random
import math
import examples.modules.PoseModule as pm
import threading
from playsound import playsound
import pandas as pd
import time
import moviepy.editor as mv
import parselmouth
import os.path
from pytube import YouTube

import matplotlib.pyplot as plt
import seaborn as sns

#################################################################################

import hydralit as hy
import hydralit_components as hc

st.set_page_config(layout='wide', page_icon="🏃‍♂️")


## 네비바 ##
app = hy.HydraApp(title='PipeRun | Home',
  favicon="🏃‍♂️",
  hide_streamlit_markers=False,
  use_banner_images=[None,None,{'header':"<h3 style='text-align:center;padding: 0px 0px;color:black;font-size:200%; font-weight:800; color:#2452c0;text-decoration:none'><a style='text-decoration:none' href='http://localhost:8501/'>PipeRun</a></h3><br>"},None,None],
  navbar_theme={'txc_inactive':'#000000', 'menu_background':'#FFFFFF', 'txc_active' : "#000000", 'option_active':'#FFFFFF'},
  )


#################################################################################

## css 관련 ##

import streamlit.components.v1 as components
import altair as alt

from typing import List, Optional

import markdown
import pandas as pd
from plotly import express as px

#  <style>
#             .navbar-mainbg {
#             background-color:#ffffff;
#             }
#             </style>


#################################################################################################

# class App():
#     global mp_drawing, mp_holistic, DEMO_IMAGE, DEMO_VIDEO, DEMO_VIDEO_RUNNING

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEMO_IMAGE = './demo/demo.png'
DEMO_VIDEO = './demo/demo.mp4'
DEMO_VIDEO_RUNNING = './demo/running.mp4'

@st.cache()
def image_resize(image, width=None, height=None, inter =cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))


    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

@app.addapp(title='Home', is_home=True)
def home():
    pygame.mixer.stop()
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # bootstrap
    components.html(
        """
        <html>

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@500&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.14.0/css/all.css" integrity="sha384-HzLeBuhoNPvSl5KYnjx0BT+WB0QEEqLprO+NBkkk5gbc67FTaL7XIGa2w1L0Xbgc" crossorigin="anonymous">
        <style>
        .main_txt{font-size:40px; font-weight:bold;}
        .main_sm_txt{line-height: calc(1.4 + var(--space) / 100); font-size:22px; margin-top : 50px;}
        .f_box{display:inline-block; margin-right :50px; width:300px; padding: 50px 20px 0 50px; background-color : #ffffff; height:300px;}
        .main-tit {text-align: center;  margin-bottom: 80px; color: #1f3ec2; font-size: 14px;   text-transform: uppercase; letter-spacing: .2em;}
        .maintxt01{font-size: 40px;color: #111;font-weight: 200;}
        .f_box_txt{font-size:20px; margin-top:20px; margin-bottom:20px; color: #111; white-space: nowrap;}
        .f_box_txt_2{font-size: 16px; color: #414141; text-overflow: ellipsis; overflow: hidden; word-wrap: break-word; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;}

        .f_main_box{padding-left:40px;padding-top:70px; width:100%; height:600px; background-color : #f8f8f8; }
        .f_main_box_2{padding-left:100px;padding-top:170px; width:100%; height:400px; background-color :#ffffff; }

        .bluepoint{border-bottom : 3px solid #b4e7f8; box-shadow:inset 0 -4px #b4e7f8;}
        .maintxt02{font-size: 22px; color: #a6a6a6; margin-top: 40px;}
        .blue{ color: #2452c0;} 
        .footer-txt{color: #fff; font-size:18px;}
        .footer-txt2{font-size:25px; font-wight:bold; color: #fff;}

        .footer-box{border-top:1px solid #222; margin-top:100px; padding:100px 50px 100px 200px; background-color: #9ea1a9;}
        .f_copy{font-size:10px; color: #fff;}

        </style>
        <body style="font-family:Noto Sans KR">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-uWxY/CJNBR+1zjPWmfnSnVxwRheevXITnMqoEIeG1LJrdI0GlVs/9cVSyPYXdcSF" crossorigin="anonymous">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-kQtW33rZJAHjgefvhyyzcGF3C5TFyBQBA13V1RKPf4uH+bwyzQxZ6CmMZHmNBEfJ" crossorigin="anonymous"></script>
            <div id="carouselExampleCaptions" class="carousel slide" data-bs-ride="carousel">
                <div class="carousel-indicators">
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="1" aria-label="Slide 2"></button>
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="2" aria-label="Slide 3"></button>
            </div>
            <div class="carousel-inner">
                <div class="carousel-item active">
                <img src="https://www.pennmedicine.org/-/media/images/miscellaneous/fitness%20and%20sports/woman_exercise_home.ashx" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>Game with AI, Try it!</h5>
                    <p>Fun ways to get 30 minutes of physical activity today</p>
                </div>
                </div>
                <div class="carousel-item">
                <img src="https://image.urbanlifehk.com/wp-content/uploads/2021/10/087802b0.jpg" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>Let's have fun exercising together</h5>
                    <p>Anywhere, Anyone, with AI</p>
                </div>
                </div>
                <div class="carousel-item">
                <img src="https://www.tonicproducts.com/wp-content/uploads/2020/05/iStock-1141568835-scaled.jpg" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>How much do you exercise every day?</h5>
                    <p>It's exercise time</p>
                </div>
                </div>
            </div>
            <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Previous</span>
            </button>
            <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Next</span>
            </button>
            </div>

            <div class="container" >
                <div class="row " style="margin-bottom:100px; margin-top:100px;">
                    <div class="col-md-6 col-xs-12" >
                        <span class="maintxt01">AI와 함께하는 <strong class="bluepoint">헬스케어 </strong></span>
                        <p class="maintxt02"> 인공신경망으로 정확한 운동 자세를 학습한 AI가 당신과 함께 합니다.<br>
                        카메라를 켜보세요. <strong class="blue"> 당신의 모든 움직임</strong>을 인식하고 알려줍니다. <br>
                        이제 당신이 학습할 차례입니다! </p>

                    </div>
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/6707079/pexels-photo-6707079.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500" class="w-100">
                    </div>
                </div>



                <div class="row">
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/4498366/pexels-photo-4498366.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500" class="w-100">
                    </div>
                    <div class="col-md-6 col-xs-12">
                        <span class="maintxt01">리듬 타고 싶은 <strong class="bluepoint">멜로디</strong>가 있나요?</span>
                        <p class="maintxt02">좋아하는 노래, TOP100, 지루한 인터넷 강의?<br>
                            소리만 있으면 무엇이든 가능합니다.<br>
                            일단 올려주세요!<br>
                            그리고 <strong class="blue">리듬에 몸을 맡기고 움직이면 됩니다.</strong>
                        </p>

                    </div>

                </div>
                
                <div class="row " style="margin-bottom:100px; margin-top:100px;">
                    <div class="col-md-6 col-xs-12" >
                        <span class="maintxt01"> 어디서든 <strong class="bluepoint">활력있게!</strong></span>
                        <p class="maintxt02"> 실내에서 가만히 있기 힘드셨죠? <br>
                        즐겁고 꾸준하게 운동해요.</p>
                    </div>
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/6707079/pexels-photo-6707079.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500" class="w-100">
                    </div>
                </div>

                </div>
                <div class="row" style="margin-top : 100px; ">
                    <div class="col-md-12 f_main_box">
                        <div class="col-md-12 text-center">
                            <p class="main-tit">Service</p>
                        </div>

                        <div class="f_box" style="margin-left:200px;">
                            <img src="https://ko.exerd.com/asset/images/ic_m0301.png">
                            <p class="f_box_txt">논리 물리 통합 모델링</p>
                            <p class="f_box_txt_2">논리와 물리 모델링을 동시 보면서 편집가능</p>
                        </div>
                        <div class="f_box">
                            <img src="https://ko.exerd.com/asset/images/ic_m0304.png">
                            <p class="f_box_txt">논리 물리 통합 모델링</p>
                            <p class="f_box_txt_2">논리와 물리 모델링을 동시 보면서 편집가능</p>
                        </div>
                        <div class="f_box">
                            <img src="https://ko.exerd.com/asset/images/ic_m0309.png">
                            <p class="f_box_txt">논리 물리 통합 모델링</p>
                            <p class="f_box_txt_2">논리와 물리 모델링을 동시 보면서 편집가능</p>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-12 f_main_box_2">
                        <p class="maintxt01" style="font-weight:bold; color: #2452c0;text-align:center;">어렵기만 한 운동 자세, 이제 걱정마세요</p>
                        <p class="maintxt02" style="text-align:center;">카메라를 켜보세요. 당신의 모든 움직임을 인식하고 알려줍니다. </p>
                    </div>
                </div>
            </div>
        </body>
        <footer class="footer-box">
            <div class="row">
                <div class="col-md-12">
                    <p class="footer-txt2">하이파이프 <i style="font-size:30px; color: #fff;" class="fas fa-running"></i></p>
                </div>
                <div class="col-md-12">
                    <p class="footer-txt">전다운, 강민지, 이국진, 전선유</p>
                </div>
                <p class="f_copy">Copyright © 2021 Pipe Run. All Rights Reserved.</p>
        </footer>
        </html>
        """,
        height=4600)

    # st.image(images/gym.jpg)



############################################################################################################################################################

# @app.addapp(title='About App')
def about_App():
    st.markdown('In this Application we are using **Mediapipe** for creating a Holistic App. **Streamlit** is to create the Web Graphical User Interface (GUI)')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )
    st.video('https://youtu.be/wyWmWaXapmI')

    st.markdown('''
                # About Our Team \n
                Hey this is ** Hi-Pipe ** from ** GIS ** \n

                If you are interested in playing game with Mediapipe. \n

                -
    ''')

############################################################################################################################################################
# @app.addapp(title='Running Detection')
def running_Detection():
    global last_action

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="models/running_modelss.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.last_action
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    seq = []
    action_seq = []
    last_action = None
    seq_length = 30
    actions = ['run', 'stop']

    video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    # Layout
    col1, col2 = st.columns(2)

    with col1:
        use_webcam = st.button('Use Webcam')
        record = st.checkbox("Record Video")


    with col2:
        my_expander = st.expander("Settings", expanded=False)
        with my_expander:
            max_faces = st.number_input('Maximum Number of Face', value=2, min_value=1)
            st.markdown('---')
            detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
            tracking_confidence = st.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
            st.markdown('---')


    st.set_option('deprecation.showfileUploaderEncoding', False)

    if record:
        st.checkbox("Recording", value=True)


    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )


    st.markdown('## Output')

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.write('')
    with col2:
        stframe = st.empty()
    with col3:
        st.write('')

    st.markdown('---')

    tffile = tempfile.NamedTemporaryFile(delete=False)

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO_RUNNING)
            tffile.name = DEMO_VIDEO_RUNNING

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording Part
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))


    st.text("Input Video")
    st.video(tffile.name)

    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)
    # kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Holistic**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    ## Dashboard
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence = detection_confidence,
        min_tracking_confidence = tracking_confidence
        ) as holistic:

        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
            i += 1
            ret, img = vid.read()
            if not ret:
                continue

            results = holistic.process(img)
            img.flags.writeable = True

            face_count = 0

            if results.pose_landmarks is not None:
                joint = np.zeros((33, 4))
                for j, lm in enumerate(results.pose_landmarks.landmark):
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

                # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                # Test model on random input data.
                # input_shape = input_details[0]['shape']
                # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

                # 시퀀스 데이터와 넘파이화
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                input_data = np.array(input_data, dtype=np.float32)

                # tflite 모델을 활용한 예측
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                y_pred = interpreter.get_tensor(output_details[0]['index'])
                i_pred = int(np.argmax(y_pred[0]))
                # conf = y_pred[i_pred]

                # if conf < 0.9:
                #     continue

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
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
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
                cv2.rectangle(img, (0,0), (250, 60), (245, 117, 16), -1)

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


            # FPS Counter logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(img)

            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

            img = cv2.resize(img, (0,0), fx=0.8, fy=0.8)
            img = image_resize(image = img, width = 640)
            stframe.image(img, channels = 'BGR', use_column_width = True)


        ########################################################
        # End mediapipe opencv logic
        ########################################################

        ## Holistic Landmark Drawing
        # for holistic_landmarks in results.face_landmarks:
        # face_count += 1



            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        # st.image(out_image, use_column_width=True)

############################################################################################################################################################

# @app.addapp(title='Run on Image')
def run_on_Image():
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    st.markdown("**Detected Faces, Hands and Pose**")
    kpi1_text = st.markdown("0")

    max_faces = st.number_input('Maximum Number of Face', value=2, min_value=1)
    st.markdown('---')
    detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.markdown('---')

    img_file_buffer = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.text('Original Image')
    st.image(image)

    face_count = 0

    ##Dashboard
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence = detection_confidence
        ) as holistic:

        results = holistic.process(image)
        out_image = image.copy()

        ## Holistic Landmark Drawing
        # for holistic_landmarks in results.face_landmarks:
        face_count += 1

        # mp_drawing.draw_landmarks(
        #     image = out_image,
        #     landmark_list = results.pose_landmarks,
        #     connections = mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec = drawing_spec
        # )
        if results.pose_landmarks:
            face_count += 1
            mp_drawing.draw_landmarks(
                out_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                out_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)


# @app.addapp(title='Run on Video')
def run_on_Video():
    global codec, out

    video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    # Layout
    col1, col2 = st.columns(2)

    with col1:
        use_webcam = st.button('Use Webcam')
        record = st.checkbox("Record Video")


    with col2:
        my_expander = st.expander("Settings", expanded=False)
        with my_expander:
            max_faces = st.number_input('Maximum Number of Face', value=2, min_value=1)
            st.markdown('---')
            detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
            tracking_confidence = st.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
            st.markdown('---')


    st.set_option('deprecation.showfileUploaderEncoding', False)

    if record:
        st.checkbox("Recording", value=True)


    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    # max_faces = st.number_input('Maximum Number of Face', value=2, min_value=1)
    # st.markdown('---')
    # detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    # tracking_confidence = st.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    # st.markdown('---')

    st.markdown('## Output')

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.write('')
    with col2:
        stframe = st.empty()
    with col3:
        st.write('')

    st.markdown('---')
    
    tffile = tempfile.NamedTemporaryFile(delete=False)

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording Part
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # codec = cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.text("Input Video")
    st.video(tffile.name)

    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)
    # kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Holistic**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    ## Dashboard
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence = detection_confidence,
        min_tracking_confidence = tracking_confidence
        ) as holistic:

        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            results = holistic.process(frame)
            frame.flags.writeable = True

            face_count = 0

            if results.pose_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # mp_drawing.draw_landmarks(
                # image = frame,
                # landmark_list = results.pose_landmarks,
                # connections = mp_holistic.POSE_CONNECTIONS,
                # landmark_drawing_spec = drawing_spec,
                # connection_drawing_spec = drawing_spec)

            # FPS Counter logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(frame)

            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)

        ########################################################
        # End mediapipe opencv logic
        ########################################################

        ## Holistic Landmark Drawing
        # for holistic_landmarks in results.face_landmarks:
        # face_count += 1



            # kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        # st.image(out_image, use_column_width=True)


# @app.addapp(title='Music')
def music():
    with st.beta_expander("Upload MP3, MP4 Files"):
        st.file_uploader("Select MP3, MP4 Files:",
                                    accept_multiple_files=True,
                                    type=['mp3','mp4'])


@app.addapp(title='Hi PipeRunner')
def piperun_selectmode_tflite_mod():

<<<<<<< Updated upstream
    use_webcam = st.button('Use Webcam')
    record = st.checkbox("Record Video")
=======
    pygame.mixer.stop()

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        font-size: 25px;
        background-color: #2452c0;
        color:#ffffff;
    }
    div.stButton > button:hover {
        font-size: 25px;
        background-color: #b4e7f8;
        color:#000000;
        border:#ffffff 1px solid;
        }
    </style>""", unsafe_allow_html=True)

    
    # record = st.checkbox("Record Video")
>>>>>>> Stashed changes

    col1, col2, col3 = st.columns([2.18, 1, 2])
    with col1:
        st.write('')
    with col2:
        use_webcam = st.button('Run Hi PipeRunner!')
    with col3:
        st.write('')

    col4, col5, col6 = st.columns([1, 3, 1])
    with col4:
        st.write('')
    with col5:
        stframe = st.empty()
    with col6:
        st.write('')

    if use_webcam:
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        # Recording Part
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        
        fps = 0

        detector = hm.HolisticDetector()
        bg_filter = sm.SegmentationFilter()

        # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        bg_color = (192, 192, 192)
        #bg_image_path = 'images/bg_night.jpg'
        bg_image_path = None

        count_text_color = (10,10,10)
        count_backgound_color = (245,245,245)

        # Load TFLite model and allocate tensors.
        interpreter_1 = tf.lite.Interpreter(model_path="models/walking_modelss.tflite")
        interpreter_1.allocate_tensors()
        interpreter_2 = tf.lite.Interpreter(model_path="models/running_modelss.tflite")
        interpreter_2.allocate_tensors()
        interpreter_3 = tf.lite.Interpreter(model_path="models/jumping_modelss.tflite")
        interpreter_3.allocate_tensors()
        interpreter_4 = tf.lite.Interpreter(model_path="models/airrope_modelss.tflite")
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


        seq = []
        action_seq = []
        last_action = None

        actions = ['fit', 'stop']
        seq_length = 30

        # default mode
        mode = "select"
        app_mode = "running"
        wording = ""
        i_pred = 1

        header_1 = overlayList[0]
        header_2 = overlayList[2]
        header_3 = overlayList[4]
        header_4 = overlayList[6]
        header_5 = overlayList[8]
        header_6 = overlayList[12]

        running_select_count = 0
        walking_select_count = 0
        jumping_select_count = 0
        airrope_select_count = 0

        # HP, kcal 초기 값
        my_HP = 100 
        total_cal = 0
        run_cal = 0
        walk_cal = 0
        jump_cal = 0
        rope_cal = 0

        sounds = {}  # 빈 딕셔너리 생성
        pygame.mixer.init()
        sounds["alaram"] = pygame.mixer.Sound("./examples/Assets/Sounds/alaram_audio.mp3")  # 재생할 파일 설정
        sounds["alaram"].set_volume(1)

        sounds["bgm"] = pygame.mixer.Sound("./examples/Assets/Sounds/ex_bgm.wav")
        sounds["bgm"].set_volume(0.2)
        sounds["bgm"].play() # bgm

        mute_count = 0
        mute_dir = 0
       
        ## Dashboard
        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
            success, img = vid.read()
            if not success:
                continue

            # 사용자 인식을 위해 Filp
            img = cv2.flip(img, 1)
            seg = img.copy()

            if bg_image_path != None:
                seg = bg_filter.Image(seg, img_path=bg_image_path)
            seg = cv2.resize(seg, (640, 480))

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
                foot_x, foot_y = lmList[28][1:3]

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
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        last_action = this_action

                # 시퀀스 데이터와 넘파이화
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                input_data = np.array(input_data, dtype=np.float32)
                
                if (540<x1<=640 and y1 < 50) or (540<x2<=640 and y2 < 50):
                    mode = "select" 
                    print("select mode")

                if mode == "exercise":
                    if app_mode == "walking":
                        print("walking mode activate")
                        interpreter_1.set_tensor(input_details[0]['index'], input_data)
                        interpreter_1.invoke()
                        y_pred = interpreter_1.get_tensor(output_details[0]['index'])
                        i_pred = int(np.argmax(y_pred[0]))

                        if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                            my_HP += 0.18
                            walk_cal += 4.0
                    elif app_mode == "running":
                        print("running mode activate")
                        interpreter_2.set_tensor(input_details[0]['index'], input_data)
                        interpreter_2.invoke()
                        y_pred = interpreter_2.get_tensor(output_details[0]['index'])
                        i_pred = int(np.argmax(y_pred[0]))

                        if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                            my_HP += 0.18
                            walk_cal += 8.0
                    elif app_mode == "jumping":
                        print("jumping mode activate")
                        interpreter_3.set_tensor(input_details[0]['index'], input_data)
                        interpreter_3.invoke()
                        y_pred = interpreter_3.get_tensor(output_details[0]['index'])
                        i_pred = int(np.argmax(y_pred[0]))

                        if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                            my_HP += 0.18
                            walk_cal += 5.5
                    elif app_mode == "air rope":
                        print("air rope mode activate")
                        cv2.line(seg, (100, 460), (540,460), (0,0,200), 2)
                        if foot_y < 460:
                            interpreter_4.set_tensor(input_details[0]['index'], input_data)
                            interpreter_4.invoke()
                            y_pred = interpreter_4.get_tensor(output_details[0]['index'])
                            i_pred = int(np.argmax(y_pred[0]))

                            if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                                my_HP += 0.18
                            walk_cal += 5.5
                        else:
                            wording = "Please Show Your Feet"
                            coords = (130, 250)
                            cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*18, coords[1]-30), (230, 230, 230), -1) 
                            cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)
                    if wording != 'Please Show Your Feet':
                        my_HP -= 0.1
                    if my_HP > 100:
                        my_HP = 100
                    if this_action.split(' ')[0] == 'stop':
                        if 30 <= my_HP <= 35 or 70 <= my_HP <= 75:
                            sounds["alaram"].play()

                    if my_HP <= 0:
                        wording = "GO! RUN! GO! RUN!"
                        coords = (160, 250)
                        cv2.rectangle(seg, (coords[0], coords[1]+5), (coords[0]+len(wording)*18, coords[1]-30), (230, 230, 230), -1)  
                        cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)
                        my_HP = 0   

                    seg[0:50, 540:640] = header_5

                    # Get status box
                    cv2.rectangle(seg, (0,0), (330, 60), (16, 117, 245), -1)
        
                    # Display Class
                    cv2.putText(seg, 'ACTION'
                            , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(seg, this_action.split(' ')[0]
                            , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(seg, 'HP'
                            , (110,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(seg, str(round(my_HP, 1))
                            , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    total_cal = walk_cal + run_cal + jump_cal + rope_cal

                    cv2.putText(seg, 'cal'
                            , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(seg, str(total_cal)
                            , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                elif mode == "select":
                    wording = "Total Calories : "
                    coords = (130, 120)
                    cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*20, coords[1]-30), (230, 230, 230), -1) 
                    cv2.putText(seg, wording + str(total_cal), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)

                    if mute_dir == 0:
                        header_6 = overlayList[12] # music_on button activate 
                    else:
                        header_6 = overlayList[10] # mute button activate 

                    header_5 = overlayList[9]
                # Checking for the click
                    if x1 < 100:
                        # mute
                        if y1<50:
                            if mute_dir == 0:
                                header_6 = overlayList[13] # music_on button activate 
                            else:
                                header_6 = overlayList[11] # mute button activate 

                            mute_count += 1
                            if (mute_count == 20) and (mute_dir == 0):
                                sounds["bgm"].stop()
                                mute_count = 0
                                mute_dir = 1
                                header_6 = overlayList[13]
                                        
                            elif (mute_count == 20) and (mute_dir == 1):
                                sounds["bgm"].play()
                                mute_count = 0
                                mute_dir = 0
                                header_6 = overlayList[11]

                        # walking 
                        if 90<=y1<190:
                            print("walking mode")
                            header_1 = overlayList[1]
                            header_2 = overlayList[2]
                            header_3 = overlayList[4]
                            header_4 = overlayList[6]
                            

                            app_mode = "walking"
                            walking_select_count += 1
                            running_select_count = 0
                            jumping_select_count = 0
                            airrope_select_count = 0
                            if walking_select_count > 10:
                                walking_select_count = 0
                                app_mode = "walking"
                                mode = "exercise"
                                header_5 = overlayList[8]

                        # running
                        elif 290<=y1<390:
                            print("running mode")
                            header_1 = overlayList[0]
                            header_2 = overlayList[3]
                            header_3 = overlayList[4]
                            header_4 = overlayList[6]

                            app_mode = "running"
                            running_select_count += 1
                            walking_select_count = 0
                            jumping_select_count = 0
                            airrope_select_count = 0
                            if running_select_count > 10:
                                running_select_count = 0
                                app_mode = "running"
                                mode = "exercise"
                                header_5 = overlayList[8]

                    elif x2 > 540:
                        # jumping
                        if 90<=y2<190:
                            print("jumping mode")
                            header_1 = overlayList[0]
                            header_2 = overlayList[2]
                            header_3 = overlayList[5]
                            header_4 = overlayList[6]

                            app_mode = "jumping"
                            jumping_select_count += 1
                            running_select_count = 0
                            walking_select_count = 0
                            airrope_select_count = 0
                            if jumping_select_count > 10:
                                jumping_select_count = 0
                                app_mode = "jumping"
                                mode = "exercise"
                                header_5 = overlayList[8]
                            
                        # air rope
                        elif 290<=y2<390:
                            print("air rope mode")
                            header_1 = overlayList[0]
                            header_2 = overlayList[2]
                            header_3 = overlayList[4]
                            header_4 = overlayList[7]

                            app_mode = "air rope"
                            airrope_select_count += 1
                            jumping_select_count = 0
                            running_select_count = 0
                            walking_select_count = 0
                            if airrope_select_count > 10:
                                airrope_select_count = 0
                                app_mode = "air rope"
                                mode = "exercise"
                                header_5 = overlayList[8]
                    
                    

                    seg[90:190, 0:100] = header_1
                    seg[290:390, 0:100] = header_2
                    seg[90:190, 540:640] = header_3
                    seg[290:390, 540:640] = header_4
                    seg[0:50, 540:640] = header_5
                    seg[0:50, 0:100] = header_6
            
                coords = tuple(np.multiply(
                                np.array(
                                    (result.pose_landmarks.landmark[7].x+20, 
                                        result.pose_landmarks.landmark[7].y))
                            , [640,480]).astype(int))
                

                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(seg)


                # seg = cv2.resize(seg, (0,0), fx=0.8, fy=0.8)
                # seg = image_resize(image = seg, width = 900)
                # stframe.image(seg, channels = 'BGR', use_column_width = 'auto')

            else:
                wording = "Please Appear On The Screen"
                coords = (80, 250)
                cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*18, coords[1]-30), (230, 230, 230), -1) 
                cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)

                # cv2.imshow("Image", seg)  

            seg = cv2.resize(seg, (0,0), fx=0.8, fy=0.8)
            seg = image_resize(image = seg, width = 900)
            stframe.image(seg, channels = 'BGR', use_column_width = 'auto')

# @app.addapp(title='Hi Challenger')
def piperun_selectmode_angle():
    detector = hm.HolisticDetector()
    bg_filter = sm.SegmentationFilter()
    
    use_webcam = st.button('Use Webcam')
    record = st.checkbox("Record Video")


    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.write('')
    with col2:
        stframe = st.empty()
    with col3:
        st.write('')

    if use_webcam:
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        # Recording Part
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        # st.text("Input Video")
        
        fps = 0



        folderPath = "examples\Header_Angle"
        myList = os.listdir(folderPath)
        # print(myList)

        # 덮어씌우는 이미지 리스트
        overlayList =[]

        # Header 폴더에 image를 상대경로로 지정
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            overlayList.append(image)

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

        count = 0
        select_count = 0
        pTime = 0
        dir = 0 


        ## Dashboard


        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
            ret, img = vid.read()
            if not ret:
                continue


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


                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(seg)

                seg = cv2.resize(seg, (0,0), fx=0.8, fy=0.8)
                seg = image_resize(image = seg, width = 900)
                stframe.image(seg, channels = 'BGR', use_column_width = 'auto')



###############################################################################################

@app.addapp(title='Hi Challenger')
def pipe_run_challenger():

    pygame.mixer.stop()
    
<<<<<<< Updated upstream
    use_webcam = st.button('Use Webcam')
    record = st.checkbox("Record Video")
=======
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        font-size: 25px;
        background-color: #2452c0;
        color:#ffffff;
    }
    div.stButton > button:hover {
        font-size: 25px;
        background-color: #b4e7f8;
        color:#000000;
        border:#ffffff 1px solid;
        }
    </style>""", unsafe_allow_html=True)

    
    # record = st.checkbox("Record Video")
>>>>>>> Stashed changes

    folderPath = "examples\Header"
    # header = cv2.imread(f'{folderPath}/mute.png')


    col1, col2, col3 = st.columns([2.18, 1, 2])
    with col1:
        st.write('')
    with col2:
        use_webcam = st.button('Run Hi Challenger!')
    with col3:
        st.write('')

    col4, col5, col6 = st.columns([1, 3, 1])
    with col4:
        st.write('')
    with col5:
        stframe = st.empty()
    with col6:
        st.write('')

    if use_webcam:
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        # Recording Part
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        
        fps = 0

        detector = hm.HolisticDetector()
        bg_filter = sm.SegmentationFilter()
            
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
<<<<<<< Updated upstream
        header_6 = overlayList[10]
=======
        header_6 = overlayList[14]
>>>>>>> Stashed changes

        total_count = 0

        squat_count = 0
        lunge_count = 0
        kneeup_count = 0
        sll_count = 0

        squat_select_count = 0
        lunge_select_count = 0
        kneeup_select_count = 0
        sll_select_count = 0

        total_cal = 0
        last_rows = pd.DataFrame([total_cal])
        
        total_cal_dic = {}

        squat_cal = 0
        lunge_cal = 0
        kneeup_cal = 0
        sll_cal = 0

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


        ## Dashboard
        prevTime = 0


        status_text = st.empty()
        chart = st.line_chart(last_rows)


        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
            success, img = vid.read()
            if not success:
                continue

            # 사용자 인식을 위해 Filp
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
                                squat_count += 0.5
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                squat_count += 0.5
                                dir = 0
                                squat_cal += 55
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
                        # Display Class
                        cv2.rectangle(seg, (0,0), (330, 60), (16, 117, 245), -1)
                        cv2.putText(seg, 'COUNT'
                                    , (110,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(int(squat_count))
                                    , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Display Probability
                        cv2.putText(seg, 'HP'
                                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(HP)
                                    , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Calorie Counting
                        cv2.putText(seg, 'cal'
                                    , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(squat_cal)
                                    , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
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
                                lunge_count += 0.5
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                lunge_count += 0.5
                                dir = 0
                                lunge_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
                        # Display Class
                        cv2.rectangle(seg, (0,0), (330, 60), (16, 117, 245), -1)
                        cv2.putText(seg, 'COUNT'
                                    , (110,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(int(lunge_count))
                                    , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Display Probability
                        cv2.putText(seg, 'HP'
                                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(HP)
                                    , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Calorie Counting
                        cv2.putText(seg, 'cal'
                                    , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(lunge_cal)
                                    , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
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
                                kneeup_count += 0.5
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                kneeup_count += 0.5
                                dir = 0
                                kneeup_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
                        # Display Class
                        cv2.rectangle(seg, (0,0), (330, 60), (16, 117, 245), -1)
                        cv2.putText(seg, 'COUNT'
                                    , (110,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(int(kneeup_count))
                                    , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Display Probability
                        cv2.putText(seg, 'HP'
                                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(HP)
                                    , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Calorie Counting
                        cv2.putText(seg, 'cal'
                                    , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(kneeup_cal)
                                    , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        

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
                                sll_count += 0.5
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                sll_count += 0.5
                                dir = 0
                                sll_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
                        # Display Class
                        cv2.rectangle(seg, (0,0), (330, 60), (16, 117, 245), -1)
                        cv2.putText(seg, 'COUNT'
                                    , (110,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(int(sll_count))
                                    , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Display Probability
                        cv2.putText(seg, 'HP'
                                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(HP)
                                    , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Calorie Counting
                        cv2.putText(seg, 'cal'
                                    , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(seg, str(sll_cal)
                                    , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        

                    seg[0:50, 540:640] = header_5


                elif mode == "select":

<<<<<<< Updated upstream
=======
                    if mute_dir == 0:
                        header_6 = overlayList[14] # music_on button activate 
                    else:
                        header_6 = overlayList[14] # mute button activate 

                    total_cal = squat_cal + lunge_cal + kneeup_cal + sll_cal

>>>>>>> Stashed changes
                    wording = "Total Calories : "
                    coords = (130, 120)
                    cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*20, coords[1]-30), (230, 230, 230), -1) 
                    cv2.putText(seg, wording + str(total_cal), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)

                    total_count = squat_count + lunge_count + kneeup_count + sll_count

                    wording = "Total Counts : "
                    coords = (130, 200)
                    cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*20, coords[1]-30), (230, 230, 230), -1) 
                    cv2.putText(seg, wording + str(int(total_count)), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)


                    header_5 = overlayList[9]
                # Checking for the click
                    if x1 < 100:
                        # walking 
                        if y1<50:
<<<<<<< Updated upstream
                            sounds["back"].stop()
                            break   
=======
                            if mute_dir == 0:
                                header_6 = overlayList[15] # music_on button activate 
                            else:
                                header_6 = overlayList[15] # mute button activate 

                            mute_count += 1
                            if (mute_count == 20) and (mute_dir == 0):
                                # sounds["back"].stop()
                                mute_count = 0
                                mute_dir = 1
                                header_6 = overlayList[15]

                                total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows

                                break

                            # elif (mute_count == 20) and (mute_dir == 1):
                            #     sounds["back"].play()
                            #     mute_count = 0
                            #     mute_dir = 0
                            #     header_6 = overlayList[11]

                            #     total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                            #     new_rows = pd.DataFrame([total_cal])
                            #     chart.add_rows(new_rows)
                            #     last_rows = new_rows
                                
                            
>>>>>>> Stashed changes
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

                                total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows

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

                                total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows

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
                                
                                total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows
                            
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
                                
                                total_count = squat_count + lunge_count + kneeup_count + sll_count
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows
                    
                    print(total_count)

                    seg[90:190, 0:100] = header_1
                    seg[290:390, 0:100] = header_2
                    seg[90:190, 540:640] = header_3
                    seg[290:390, 540:640] = header_4
                    seg[0:50, 540:640] = header_5
                    seg[0:50, 0:100] = header_6
                    # 테스트좀 네!
                    

                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if HP > 100:
                    HP = 100

                if record:
                    out.write(seg)

            else:
                wording = "Please Appear On The Screen"
                coords = (80, 250)
                cv2.rectangle(seg,(coords[0], coords[1]+5), (coords[0]+len(wording)*18, coords[1]-30), (230, 230, 230), -1) 
                cv2.putText(seg, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)

                # cv2.imshow("Image", seg)  

            seg = cv2.resize(seg, (0,0), fx=0.8, fy=0.8)
            seg = image_resize(image = seg, width = 900)
            stframe.image(seg, channels = 'BGR', use_column_width = 'auto')



        counts = ['Squat', 'Lunge', 'Knee Up', 'Side Lateral Raise']
        counts_data = [squat_count, lunge_count, kneeup_count, sll_count]
        counts_df = pd.DataFrame(data = map(int, counts_data), columns=['Count'], index=['Squat', 'Lunge', 'Knee Up', 'Side Lateral Raise'])
    
        colors = sns.color_palette('hls', len(counts))

        fig1 = plt.figure()
        plt.bar(counts, counts_data, color=colors)

        cal_data = [total_cal]
        cal_df = pd.DataFrame(data = map(int, cal_data), columns=['Exercise'], index=['Total Calories'])

        food_calories = {
            '밥 한공기' : 310,
            '떡볶이' : 300,
            '삼겹살' : 460,
            '라떼' : 180,
            '피자' : 404,
            '치킨' : 249,
            '초코바' : 240,
            '초밥' : 179,
            '잔치국수' : 447,
            '아이스크림' : 186,
            '햄버거 세트' : 956,
            '짜장면' : 785,
            '짬뽕' : 464,
            '국밥' : 470,
            '아메리카노' : 4,
            '사과' : 57,
            '우유' : 65,
            '바나나' : 93,
            '군고구마' : 124,
            '방울토마토' : 2,
            '두부' : 88,
            '삶은 달걀' : 68,
            '인절미' : 220,
            '치즈 케이크' : 265,
            '롤케이크' : 244,
            '슈크림빵' : 220,
            '앙버터' : 674,
            '바게트' : 44,
            '소프트콘' : 145,
            '회' : 136,
            '짜장면 + 짬뽕' : 1249,
            '지방' : 2000,
            '신난다! 이만큼' : 3000,
            '예! 전부 다' : 999999999
        }

        new_food_calories = sorted(food_calories.items(), key=lambda x:x[1], reverse=False)

        st.empty()
        st.write("---")
        st.empty()

        for food, cal in new_food_calories:
            if total_cal == 0:
                st.title(f'🥳 숨쉬기 운동이라도 하면 됐죠! 💪')
                break

            elif total_cal < cal:
                st.title("🎉 WoW~ 어디서 좀 노셨군요? 🎉")
                st.title(f'🥳 {food} 격파했어요! 빠샤! 💪')
                break

        st.balloons()

        col_df_1, col_df_2 = st.columns(2)

        with col_df_1:
            st.write(counts_df.head())

        with col_df_2:
            st.pyplot(fig1)

@app.addapp(title='Hi Clicker')
def clicker():
<<<<<<< Updated upstream
=======

    pygame.mixer.stop()

    mute_count = 0
    mute_dir = 0

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

>>>>>>> Stashed changes
    global score, x_enemy, y_enemy, count

    # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
<<<<<<< Updated upstream
    
    record = st.checkbox("Record Video")

    folderPath = "examples\Header"
=======

    # record = st.checkbox("Record Video")

    folderPath = "examples/Header"
    myList = os.listdir(folderPath)

    # 덮어씌우는 이미지 리스트
    overlayList = []

    # Header 폴더에 image를 상대경로로 지정
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
>>>>>>> Stashed changes

    header = cv2.imread(f'{folderPath}/mute.png')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write('\n')
    with col2:
        level = st.slider("레벨을 선택하세요.", 1, 5, 1)
    with col3:
        st.write('')

<<<<<<< Updated upstream

    yt_url = st.text_input('유튜브 링크를 붙여 넣어주세요')
=======
    col4, col5, col6 = st.columns([1, 3, 1])
    with col4:
        st.write('\n')
    with col5:
        stframe = st.empty()
        yt_url = st.text_input('유튜브 [공유 링크] 를 붙여 넣어주세요.')
        st.text("데모를 시행시키려면 'demo'를 입력하세요.")
        st.text("소리를 끄려면 'MUTE' 버튼을 2초간 Click! 하세요.")
    with col6:
        st.write('')

    # 노트 나오는 속도 조절 값이 줄어들 수록 생성 속도가 빨라짐
    if level == 1:
        difficulty = 40
    if level == 2:
        difficulty = 35
    if level == 3:
        difficulty = 30
    if level == 4:
        difficulty = 25
    if level == 5:
        difficulty = 20

    
>>>>>>> Stashed changes

    # if use_webcam:

    if yt_url:
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        # 유튜브 링크 입력으로 다운받아오는 부분
        download_path = './media'

<<<<<<< Updated upstream
        if not os.path.exists(download_path):
            os.makedirs(download_path)
=======
            video = YouTube(yt_url)
            video_type = video.streams.filter(progressive=True, file_extension="mp4").get_highest_resolution()
            video_type.download(video_download_path)
>>>>>>> Stashed changes

        video = YouTube(yt_url)
        video_type = video.streams.filter(progressive = True, file_extension = "mp4").get_highest_resolution()
        video_type.download('./media/')

        # 다운받은 비디오의 이름을 video.mp4로 바꾸기 -> 이러면 노래 하나만 있어야한다는 문제점.
        file_list = os.listdir(download_path)
        old_name = os.path.join(download_path, file_list[0])
        new_name = os.path.join(download_path, "video.mp4")

        os.rename(old_name, new_name)

        # mp4를 mp3 로 전환
        clip = mv.VideoFileClip("media/video.mp4")
        clip.audio.write_audiofile("media/audio.mp3")

        audio_file_path = "media/audio.mp3"
        #ㅈ
        Sound = parselmouth.Sound(audio_file_path)

        # 1초단위로
        formant = Sound.to_formant_burg(time_step=1)
        # Pitch값 추출
        pitch = Sound.to_pitch()
        df = pd.DataFrame({"times": formant.ts()})

<<<<<<< Updated upstream
        times =[]
        times.append(formant.ts())
=======
            times = []
            times.append(formant.ts())
>>>>>>> Stashed changes

        df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))

        df.to_csv("media/pitchdata.csv")

        sound = pd.read_csv(data_file_path, index_col=0)

<<<<<<< Updated upstream

        sound = pd.read_csv("media/pitchdata.csv", index_col=0)
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
=======
        df = sound['F0(pitch)']
        # sound['times'] = sound['times'].astype(int)
        normal_df = (df - df.min()) / (df.max() - df.min())
        pitch = normal_df.fillna(2)
        sound['F0(pitch)'] = pitch
>>>>>>> Stashed changes

        # 노래 삽입
        sounds = {}  # 빈 딕셔너리 생성
        pygame.mixer.init()
        sounds["slap"] = pygame.mixer.Sound("examples\Assets\Sounds\slap.wav")  # 재생할 파일 설정
        sounds["slap"].set_volume(1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
        sounds["screaming"] = pygame.mixer.Sound("examples\Assets\Sounds\Effect.wav")  # 재생할 파일 설정
<<<<<<< Updated upstream
        sounds["screaming"].set_volume(1)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
        sounds["song"] = pygame.mixer.Sound("media/audio.mp3")  # 재생할 파일 설정
        sounds["song"].set_volume(0.3)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
        
=======
        sounds["screaming"].set_volume(0.4)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
        sounds["song"] = pygame.mixer.Sound(audio_file_path)  # 재생할 파일 설정
        sounds["song"].set_volume(0.2)  # 볼륨 설정 SOUNDS_VOLUME 0~1 사이의 값을 넣으면 됨
>>>>>>> Stashed changes

        # class 객체 생성
        detector = hm.HolisticDetector()
        movedetector = pm.poseDetector()

        cal = 0
        score = 0
<<<<<<< Updated upstream
        note_count = 20
=======
        note_count = difficulty
>>>>>>> Stashed changes
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
            elif i_pitch >= 0.3 and i_pitch < 0.6:
                result = random.randint(220, 310)
            elif i_pitch >= 0.6 and i_pitch <= 1:
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


        # video = cv2.VideoCapture(0)

        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        while vid.isOpened():
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
            ret, img = vid.read()
            if not ret:
                continue

            image = img.copy()
            movedetector.findPose(image)
            footlmList = movedetector.findPosition(image, draw=False)

<<<<<<< Updated upstream
            if note_count == 10:
                mp3_time +=1
=======
            if note_count == difficulty:
                mp3_time += 1
>>>>>>> Stashed changes

            # 원형 노트 생성
            if note_count > 0 and y_enemy < 220:
                note_count -= 1
                cv2.circle(image, (x_enemy, y_enemy), 25,
                           (0, 0, random.randint(0, 255)), 5)
            # 삼각형 노트 생성
            elif note_count > 0 and y_enemy >= 220 and y_enemy < 300:
                note_count -= 1
                pts = np.array([[x_enemy, y_enemy], [x_enemy + 60, y_enemy], [x_enemy + 30, y_enemy - 49]], np.int32)
                cv2.polylines(image, [pts], True, (0, 0, random.randint(0, 255)), 3)
            # 사각형 노트 생성
            elif note_count > 0 and y_enemy >= 300:
                note_count -= 1
                cv2.rectangle(image, (x_enemy, y_enemy), (x_enemy + 40, y_enemy + 40),
                              (0, 0, random.randint(0, 255)), 5)

            # 사용자가 감지를 못 했을 때
            if note_count == 0:
                x_enemy = random.randint(50, 600)
                y_enemy = make_pitch(sound, mp3_time)
<<<<<<< Updated upstream
                note_count = 20  # random.randint(30,40)


            # image = detector.findHolistic(image, draw=False)

            # # 왼손 좌표 리스트
            # LefthandLandmarkList = detector.findLefthandLandmark(image)
            # # 오른손 좌표 리스트
            # RighthandLandmarkList = detector.findRighthandLandmark(image)
=======
                note_count = difficulty  # random.randint(30,40)
>>>>>>> Stashed changes

            if len(footlmList) != 0:

                # 음악 종료
                x1, y1 = footlmList[20][1:3]
                x2, y2 = footlmList[19][1:3]

<<<<<<< Updated upstream
                if (540<x1<=640 and y1 < 50) or (540<x2<=640 and y2 < 50):
                    print("stop command")
                    sounds["song"].stop()
                    break
=======
                # 에어 커서
                cv2.circle(image, (x1, y1), 5, (0, 0, 0), 2)
                cv2.circle(image, (x2, y2), 5, (0, 0, 0), 2)

                if (0 < x1 <= 100 and 0 < y1 <= 50) or (0 < x2 <= 100 and 0 < y2 <= 50):
                    mute_count += 1

                    if mute_dir == 0:
                        header_1 = overlayList[13]
                    else:
                        header_1 = overlayList[11]

                    if mute_count == 20 and mute_dir == 0:
                        print("stop command")
                        sounds["song"].stop()
                        mute_dir = 1
                        mute_count = 0
                    elif mute_count == 20 and mute_dir == 1:
                        print("music start")
                        sounds["song"].play()
                        mute_dir = 0
                        mute_count = 0
                else:
                    if mute_dir == 0:
                        mute_count = 0
                        header_1 = overlayList[12]
                    else:
                        mute_count = 0
                        header_1 = overlayList[10]
>>>>>>> Stashed changes

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
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # 왼쪽 무릎으로 감지 했을 때
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # 발로 감지 했을 때
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # 발로 감지 했을 때
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # 손으로 감지 했을 때
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.3
                    # 손으로 감지 했을 때
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.3
                # 삼각형일 때
                elif y_enemy >= 220 and y_enemy < 310:
                    # 오른쪽 무릎으로 감지 했을 때
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.4
                    # 왼쪽 무릎으로 감지 했을 때
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.4
                    # 발로 감지 했을 때
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # 발로 감지 했을 때
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # 손으로 감지 했을 때
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                    # 손으로 감지 했을 때
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                # 사각형일 때
                elif y_enemy >= 310:
                    # 오른쪽 무릎으로 감지 했을 때
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # 왼쪽 무릎으로 감지 했을 때
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # 발로 감지 했을 때
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.35
                    # 발로 감지 했을 때
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.35
                    # 손으로 감지 했을 때
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                    # 손으로 감지 했을 때
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # 음원 재생
                        note_count = 15
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3


                image = cv2.flip(image, 1)

                image[0:50, 0:100] = header

                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 255)
<<<<<<< Updated upstream
                text = cv2.putText(image, "Score", (480, 30), font, 1, color, 4, cv2.LINE_AA)
                text = cv2.putText(image, str(score), (590, 30), font, 1, color, 4, cv2.LINE_AA)

=======


                # Display Class
                cv2.rectangle(image, (470,0), (640, 60), (16, 117, 245), -1)
                cv2.putText(image, 'CAL'
                            , (560,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(cal)
                            , (560,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'SCORE'
                            , (485,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(score)
                            , (485,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



            else:
                wording = "Please Appear On The Screen"
                coords = (80, 250)
                cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(wording) * 18, coords[1] - 30),
                              (230, 230, 230), -1)
                cv2.putText(image, wording, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2, cv2.LINE_AA)
>>>>>>> Stashed changes

                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

<<<<<<< Updated upstream
                if record:
                    out.write(image)
                                        
                image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)
                image = image_resize(image = image, width = 900)
                stframe.image(image, channels = 'BGR', use_column_width = 'auto')
=======
            # if record:
            #     out.write(image)

            image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            image = image_resize(image=image, width=900)
            stframe.image(image, channels='BGR', use_column_width='auto')
>>>>>>> Stashed changes


app.run()


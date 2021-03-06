from re import T
from google.protobuf.symbol_database import Default
from mediapipe.python.solutions.face_mesh import FACE_CONNECTIONS
from numpy.core.fromnumeric import size
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import ImageFont, ImageDraw, Image
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

st.set_page_config(layout='wide', page_icon="πββοΈ")


## λ€λΉλ° ##
app = hy.HydraApp(title='PipeRun | Home',
  favicon="πββοΈ",
  hide_streamlit_markers=False,
  use_banner_images=[None,None,{'header':"<h3 style='text-align:center;padding: 0px 0px;color:black;font-size:200%; font-weight:800; color:#2452c0;text-decoration:none'><a style='text-decoration:none' href='http://localhost:8501/'>PipeRun</a></h3><br>"},None,None],
  navbar_theme={'txc_inactive':'#000000', 'menu_background':'#FFFFFF', 'txc_active' : "#000000", 'option_active':'#FFFFFF'},
  )


#################################################################################

## css κ΄λ ¨ ##

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
                        <span class="maintxt01">AIμ ν¨κ»νλ <strong class="bluepoint">ν¬μ€μΌμ΄ </strong></span>
                        <p class="maintxt02"> μΈκ³΅μ κ²½λ§μΌλ‘ μ΄λ μμΈλ₯Ό νμ΅ν AIκ°<br>λΉμ κ³Ό ν¨κ» ν©λλ€.<br><br>
                        <strong class="blue"> λΉμ μ μμ§μ</strong>μ μΈμνκ³  μ΄λλμ μλ €μ€λλ€. <br>
                        λ€μν λμμ΄ μΈμ κ°λ₯ν λ°λλ¨Έμ μ μ¦κ²¨λ³΄μΈμ!<br><br>
                        <strong class="blue"><strong class="bluepoint">Hi PipeRunner</strong></strong></p>

                    </div>
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/8033089/pexels-photo-8033089.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=210&w=500" class="w-100">
                    </div>
                </div>



                <div class="row">
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/8173441/pexels-photo-8173441.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500" class="w-100">
                    </div>
                    <div class="col-md-6 col-xs-12">
                        <span class="maintxt01"> νλ  λμμ <strong class="bluepoint">μ¬λ―Έμκ²!</strong></span>
                        <p class="maintxt02"> μ½κ² ν¬κΈ° μ¬μ΄ ννΈλ μ΄λ? <strong class="bluepoint">No!</strong><br>
                        ν¬κΈ°νμ§ μκ³  μ΄λν  μ μλλ‘ λμμ€μ. <br><br>
                        μ΄λμ νμ§ μμΌλ©΄ κ·μ¬μ΄ κ·μ  μΉκ΅¬λ€μ΄ λ±μ₯ν΄μ!<br>
                        <strong class="blue">μ΄μ μμ§μ¬μ λλ§μ³μΌ ν΄μ!</strong> <br><br>
                        <strong class="blue"><strong class="bluepoint">Hi Challenger</strong></strong></p>
                        

                    </div>

                </div>
                
                <div class="row " style="margin-bottom:100px; margin-top:100px;">
                    <div class="col-md-6 col-xs-12" >
                        <span class="maintxt01">λ¦¬λ¬ νκ³  μΆμ <strong class="bluepoint">λ©λ‘λ</strong>κ° μλμ?</span>
                        <p class="maintxt02">μ’μνλ λΈλμ λ§μΆ° <strong class="bluepoint">Click!</strong><br>
                            μ λλ λΈλμ λ§μΆ° <strong class="bluepoint">Click!</strong><br><br>
                            
                            λͺ¨μμ λ§μΆ° ν°μΉ ν¨λμ <strong class="bluepoint">Click!</strong> νλ©΄ μ μκ° μ¬λΌκ°μ!<br>
                            <strong class="blue">λ¦¬λ¬μ λͺΈμ λ§‘κΈ°κ³  μμ§μ΄λ©΄ λ©λλ€.<br><br>
                            
                            <strong class="bluepoint">Hi Clicker</strong></strong>
                    </div>
                    <div class="col-md-6 col-xs-12">
                        <img src="https://images.pexels.com/photos/4498366/pexels-photo-4498366.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500" class="w-100">
                    </div>
                </div>

                </div>
                <div class="row" style="margin-top : 100px; ">
                    <div class="col-md-12 f_main_box">
                        <div class="col-md-12 text-center">
                            <p class="main-tit">Service</p>
                        </div>
                        <div class="f_box" style="margin-left:100px;">
                            <img src="https://ko.exerd.com/asset/images/ic_m0305.png">
                            <p class="f_box_txt">LSTM μΈκ³΅μ§λ₯ λͺ¨λΈλ§</p>
                            <p class="f_box_txt_2">μ μλ¦¬ κ±·κΈ° λ° μ μλ¦¬ λ¬λ¦¬κΈ° λ± λμ  λμ μΈμ</p>
                        </div>
                        <div class="f_box">
                            <img src="https://ko.exerd.com/asset/images/ic_m0301.png">
                            <p class="f_box_txt">μ΄λλ κ·Έλν μκ°ν</p>
                            <p class="f_box_txt_2">μλͺ¨λ μΉΌλ‘λ¦¬ μ λ³΄λ₯Ό κ·Έλνλ‘ μκ°ννμ¬ μ κ³΅</p>
                        </div>
                        <div class="f_box">
                            <img src="https://ko.exerd.com/asset/images/ic_m0304.png">
                            <p class="f_box_txt">μ νλΈ μμκ³Ό μ°λ</p>
                            <p class="f_box_txt_2">μλμ΄μ λ°λ₯Έ ν°μΉ ν¨λ μμ±κ³Ό ν₯λ―Έ μ κ³΅</p>
                        </div>
                        
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-12 f_main_box_2">
                        <p class="maintxt01" style="font-weight:bold; color: #2452c0;text-align:center;">μ§λ£¨ν ν νΈλ μ΄λ, μ΄μ  νμ΄νλ°κ³Ό ν¨κ»νμΈμ</p>
                        <p class="maintxt02" style="text-align:center;">μΉ΄λ©λΌλ₯Ό μΌλ³΄μΈμ. μ¬λ°κ³  μ λκ² AI ννΈλ μ΄λ κ²μμ μ¦κ²¨λ³΄μΈμ. </p>
                    </div>
                </div>
            </div>
        </body>
        <footer class="footer-box">
            <div class="row">
                <div class="col-md-12">
                    <p class="footer-txt2">νμ΄νμ΄ν <i style="font-size:30px; color: #fff;" class="fas fa-running"></i></p>
                </div>
                <div class="col-md-12">
                    <p class="footer-txt">μ λ€μ΄, κ°λ―Όμ§, μ΄κ΅­μ§, μ μ μ </p>
                </div>
                <p class="f_copy">Copyright Β© 2021 Pipe Run. All Rights Reserved.</p>
        </footer>
        </html>
        """,
        height=3680)

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
@app.addapp(title='Hi PipeRunner')
def piperun_selectmode_tflite_mod():

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

        fontpath = "fonts/CarterOne-Regular.ttf"
        font = ImageFont.truetype(fontpath, 35)

        folderPath = "examples\Header_Piperunner"
        myList = os.listdir(folderPath)
        # print(myList)

        # λ?μ΄μμ°λ μ΄λ―Έμ§ λ¦¬μ€νΈ
        overlayList =[]

        # Header ν΄λμ imageλ₯Ό μλκ²½λ‘λ‘ μ§μ 
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

        hp_header = overlayList[14]
        cal_header = overlayList[15]
        hp_cal_header = overlayList[16]

        running_select_count = 0
        walking_select_count = 0
        jumping_select_count = 0
        airrope_select_count = 0

        combo_display_count = 0
        combo_count = 0
        combo = 0

        word_display_count = 0

        # HP, kcal μ΄κΈ° κ°
        my_HP = 100 
        total_cal = 0
        run_cal = 0
        walk_cal = 0
        jump_cal = 0
        rope_cal = 0

        sounds = {}  # λΉ λμλλ¦¬ μμ±
        pygame.mixer.init()
        sounds["alaram"] = pygame.mixer.Sound("./examples/Assets/Sounds/alaram_audio.mp3")  # μ¬μν  νμΌ μ€μ 
        sounds["alaram"].set_volume(0.5)

        sounds["bgm"] = pygame.mixer.Sound("./examples/Assets/Sounds/ex_bgm.wav")
        sounds["bgm"].set_volume(0.2)
        sounds["bgm"].play() # bgm

        sounds["pose_ok"] = pygame.mixer.Sound("examples\Assets\Sounds\pose_ok.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["pose_ok"].set_volume(0.5)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["get_score"] = pygame.mixer.Sound("examples\Assets\Sounds\get_score.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["get_score"].set_volume(0.5)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨

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

            if my_HP < 30:
                hp_color = (0, 0, np.random.randint(180,200))
            elif my_HP < 60:
                hp_color = (0, np.random.randint(180,200), 200)
            else:
                hp_color = (0, np.random.randint(180,200), 0)

            # μ¬μ©μ μΈμμ μν΄ Filp
            img = cv2.flip(img, 1)
            seg = img.copy()

            if bg_image_path != None:
                seg = bg_filter.Image(seg, img_path=bg_image_path)
            seg = cv2.resize(seg, (640, 480))

            # 2. Find Hand Landmarks
            # detector λΌλ ν΄λμ€λ₯Ό μ μΈνκ³  μμ μ°Ύλλ€.
            result = detector.findHolisticwithResults(img)
            # μμ μ’νλ₯Ό lmListμ μ μ₯νλ€.( μμ 21κ°μ μ’νλ₯Ό ν¬ν¨ )
            lmList = detector.findPoseLandmark(img)

            # μμ΄ κ°μ§κ° λμμ λ
            if len(lmList) != 0:
                # print(lmList)

                # tip of Index fingers(κ²μ§ μκ°λ½μ μ’ν)
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

                # μνμ€ λ°μ΄ν°μ λνμ΄ν
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
                            combo_count += 1
                    elif app_mode == "running":
                        print("running mode activate")
                        interpreter_2.set_tensor(input_details[0]['index'], input_data)
                        interpreter_2.invoke()
                        y_pred = interpreter_2.get_tensor(output_details[0]['index'])
                        i_pred = int(np.argmax(y_pred[0]))

                        if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                            my_HP += 0.18
                            run_cal += 8.0
                            combo_count += 1
                    elif app_mode == "jumping":
                        print("jumping mode activate")
                        interpreter_3.set_tensor(input_details[0]['index'], input_data)
                        interpreter_3.invoke()
                        y_pred = interpreter_3.get_tensor(output_details[0]['index'])
                        i_pred = int(np.argmax(y_pred[0]))

                        if this_action.split(' ')[0] == 'fit' and round(y_pred[0][np.argmax(y_pred[0])]) >= 0.5:
                            my_HP += 0.18
                            jump_cal += 5.5
                            combo_count += 1
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
                                rope_cal += 5.5
                                combo_count += 1
                        else:
                            wording = "Please Show Your Feet"
                            coords = (110, 170)
                            cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+407, coords[1]+52), (0, 0, 0), 2)
                            cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+405, coords[1]+50), (255, 255, 255), -1) 
                            img_pil = Image.fromarray(seg)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                            seg = np.array(img_pil)

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
                        cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+327, coords[1]+52), (0, 0, 0), 2)
                        cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+325, coords[1]+50), (255, 255, 255), -1) 
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                        seg = np.array(img_pil)
                        my_HP = 0   

                    total_cal = walk_cal + run_cal + jump_cal + rope_cal
                    if combo_count == 40:
                        combo += 1
                        combo_display_count = 20
                        combo_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                        wording = f'{int(combo)} Combo!'
                        combo_count = 0
                        sounds["get_score"].play()
                    elif combo_count == 20:
                        word_display_count = 20
                        word_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                        wording_list = ['Good!', 'Nice!', 'Fit!', 'Excellent!', 'Keep Going!']
                        wording = f'{random.choice(wording_list)}'
                        sounds["pose_ok"].play()
                        combo_count += 1
                        

                    # Get status box
                    if my_HP > 0:
                        cv2.rectangle(seg, (40, 8), (32 + int(my_HP)*3, 28), hp_color, cv2.FILLED)
                    cv2.rectangle(seg, (38, 7), (332, 29), (255, 255, 255), 2)
                    
                    if total_cal > 0:
                        cv2.rectangle(seg, (40, 40), (40 + int(total_cal * 0.02)*3, 60), (16, 117, np.random.randint(230,255)), cv2.FILLED)
                    cv2.rectangle(seg, (38, 39), (332, 61), (255, 255, 255), 2)

                    # combo display
                    if combo_display_count > 0 :
                        combo_color = (0,np.random.randint(200, 240),np.random.randint(200, 240),0)
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(combo_coords, f'{wording}', font=font, fill=combo_color)
                        seg = np.array(img_pil)
                        combo_display_count -= 1

                    if word_display_count > 0 :
                        word_color = (np.random.randint(180, 255),0,0,0)
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(word_coords, f'{wording}', font=font, fill=word_color)
                        seg = np.array(img_pil)
                        word_display_count -= 1

                    seg[0:50, 540:640] = header_5
                    # seg[5:35, 5:35] = hp_header
                    # seg[40:70, 5:35] = cal_header
                    seg[3:69, 5:35] = hp_cal_header

                elif mode == "select":
                    wording = f"Total Calories : {int(total_cal*0.001)}"
                    coords = (70, 215)
                    cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+507, coords[1]+52), (0, 0, 0), 2)
                    cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+505, coords[1]+50), (255, 255, 255), -1) 
                    img_pil = Image.fromarray(seg)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                    seg = np.array(img_pil)
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
                    
            
                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

            else:
                wording = "Please Appear On The Screen"
                coords = (50, 220)
                cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+507, coords[1]+52), (0, 0, 0), 2)
                cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+505, coords[1]+50), (255, 255, 255), -1) 
                img_pil = Image.fromarray(seg)
                draw = ImageDraw.Draw(img_pil)
                draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                seg = np.array(img_pil)

            seg = cv2.resize(seg, (0,0), fx=0.8, fy=0.8)
            seg = image_resize(image = seg, width = 900)
            stframe.image(seg, channels = 'BGR', use_column_width = 'auto')


###############################################################################################

@app.addapp(title='Hi Challenger')
def pipe_run_challenger():

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

        fontpath = "fonts/CarterOne-Regular.ttf"
        font = ImageFont.truetype(fontpath, 35)

        folderPath = "examples/Header_Challenger"
        backgroundfolderPath = "examples/background"
        myList = os.listdir(folderPath)
        background_myList = os.listdir(backgroundfolderPath)
        # print(background_myList)

        # λ?μ΄μμ°λ μ΄λ―Έμ§ λ¦¬μ€νΈ
        overlayList =[]
        backgroundList = []

        # Header ν΄λμ imageλ₯Ό μλκ²½λ‘λ‘ μ§μ 
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            overlayList.append(image)

        for imPath in background_myList:
            image_path = f'{backgroundfolderPath}/{imPath}'
            backgroundList.append(image_path)
        # print(backgroundList)

        # λΉλμ€ μΈν
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
        header_6 = overlayList[14]

        hp_header = overlayList[16]
        cal_header = overlayList[17]
        hp_cal_header = overlayList[18]

        total_count = 0

        combo_display_count = 0
        word_display_count = 0
        good_wording = ''

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


        mute_count = 0
        mute_dir = 0

        # sound
        sounds = {}  # λΉ λμλλ¦¬ μμ±
        pygame.mixer.init()
        sounds["pose_ok"] = pygame.mixer.Sound("examples\Assets\Sounds\pose_ok.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["pose_ok"].set_volume(0.5)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["get_score"] = pygame.mixer.Sound("examples\Assets\Sounds\get_score.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["get_score"].set_volume(0.5)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["click"] = pygame.mixer.Sound("examples\Assets\Sounds\click.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["click"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["back"] = pygame.mixer.Sound("examples/Assets/Sounds/back.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["back"].set_volume(0.1)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["back"].play()


        sounds["g1"] = pygame.mixer.Sound("examples\Assets\Sounds\g1.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g1"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["g2"] = pygame.mixer.Sound("examples\Assets\Sounds\g2.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g2"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["g3"] = pygame.mixer.Sound("examples\Assets\Sounds\g3.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g3"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["g4"] = pygame.mixer.Sound("examples\Assets\Sounds\g4.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g4"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["g5"] = pygame.mixer.Sound("examples\Assets\Sounds\g5.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g5"].set_volume(0.3)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["g6"] = pygame.mixer.Sound("examples\Assets\Sounds\g6.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["g6"].set_volume(0.5)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨


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

            # μ¬μ©μ μΈμμ μν΄ Filp
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
                hp_color = (0, 0, np.random.randint(180,200))
                bg_image_path = backgroundList[4]
            elif HP == 30:
                sounds["g3"].play()
            elif HP < 45:
                bg_image_path = backgroundList[3]
            elif HP == 45:        
                sounds["g2"].play()
            elif HP < 60:
                hp_color = (0, np.random.randint(180,200), 200)
                bg_image_path = backgroundList[2]
            elif HP == 60: 
                sounds["g1"].play()
            elif HP < 75:
                bg_image_path = backgroundList[1]
            elif HP == 75:
                sounds["g6"].play()
            else:
                hp_color = (0, np.random.randint(180,200), 0)
                bg_image_path = backgroundList[0]




            if bg_image_path != None:
                seg = bg_filter.Image(seg, img_path=bg_image_path)
            seg = cv2.resize(seg, (640, 480))

            img = detector.findHolistic(img, draw=False)
            pose_lmList = detector.findPoseLandmark(img, draw=False)

            # μ μ²΄ κ°μ§κ° λμμ λ
            if len(pose_lmList) != 0:

                right_hand_x = pose_lmList[16][1]
                left_hand_x = pose_lmList[15][1]
                right_shoulder_x = pose_lmList[12][1]
                left_shoulder_x = pose_lmList[11][1]

                x1, y1 = pose_lmList[19][1:3]
                x2, y2 = pose_lmList[20][1:3]
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
                        coords = (200, 280)
                        cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+212, coords[1]+52), (0, 0, 0), 2)
                        cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+210, coords[1]+50), (255, 255, 255), -1) 
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((200, 280), f'{wording}', font=font, fill=(0,0,0,0))
                        seg = np.array(img_pil)

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
                                per = np.interp(total_angle, (70, 300), (0, 100))
                                bar = np.interp(total_angle, (70, 300), (450, 100))
                                
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
                                word_display_count = 20
                                word_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                wording_list = ['Good!', 'Nice!', 'Fit!', 'Excellent!', 'Keep Going!']
                                wording = f'{random.choice(wording_list)}'
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                squat_count += 0.5
                                combo_display_count = 20
                                combo_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                dir = 0
                                squat_cal += 55
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                        
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
                                word_display_count = 20
                                word_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                wording_list = ['Good!', 'Nice!', 'Fit!', 'Excellent!', 'Keep Going!']
                                wording = f'{random.choice(wording_list)}'
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                lunge_count += 0.5
                                combo_display_count = 20
                                combo_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                dir = 0
                                lunge_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
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

                        per = np.interp(total_angle, (170, 200), (0, 100))
                        bar = np.interp(total_angle, (170, 200), (450, 100))
                        
                        
                        # Check for the curls
                        color = (0,0,200)
                        if per == 100:
                            color = (0,200,0)        
                            if dir == 0:
                                sounds["pose_ok"].play()
                                kneeup_count += 0.5
                                word_display_count = 20
                                word_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                wording_list = ['Good!', 'Nice!', 'Fit!', 'Excellent!', 'Keep Going!']
                                wording = f'{random.choice(wording_list)}'
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                kneeup_count += 0.5
                                combo_display_count = 20
                                combo_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                dir = 0
                                kneeup_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)        
                
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
                                word_display_count = 20
                                word_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                wording_list = ['Good!', 'Nice!', 'Fit!', 'Excellent!', 'Keep Going!']
                                good_wording = f'{random.choice(wording_list)}'
                                dir = 1
                                HP += difficulty
                        if per == 0:
                            color = (0,200,0)        
                            if dir == 1:
                                sounds["get_score"].play()
                                sll_count += 0.5
                                combo_display_count = 20
                                combo_coords = (np.random.randint(50, 300), np.random.randint(150, 300))
                                dir = 0
                                sll_cal += 33
                                HP += difficulty

                        # Draw bar
                        cv2.rectangle(seg, (570, 100), (610, 450), color, 3)
                        cv2.rectangle(seg, (570, int(bar)), (610, 450), color, cv2.FILLED)
                        cv2.putText(seg, f'{int(per)}%', (565, 80),
                                    cv2.LINE_AA, 0.8, color, 2)   

                    total_count = squat_count + lunge_count + kneeup_count + sll_count 
                    total_cal = squat_cal + lunge_cal + kneeup_cal + sll_cal    

                    
                    if HP > 0:
                        cv2.rectangle(seg, (40, 8), (40 + int(HP)*3, 28), hp_color, cv2.FILLED)
                    cv2.rectangle(seg, (38, 7), (332, 29), (255, 255, 255), 2)
                    
                    if total_cal > 0:
                        cv2.rectangle(seg, (40, 40), (40 + int(total_cal * 0.1)*3, 60), (16, 117, np.random.randint(230,255)), cv2.FILLED)
                    cv2.rectangle(seg, (38, 39), (332, 61), (255, 255, 255), 2)

                    # combo display
                    if combo_display_count > 0 :
                        combo_color = (0,np.random.randint(200, 240),np.random.randint(200, 240),0)
                        wording = f'{int(total_count)} Combo!'
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(combo_coords, f'{wording}', font=font, fill=combo_color)
                        seg = np.array(img_pil)
                        combo_display_count -= 1

                    if word_display_count > 0 :
                        word_color = (np.random.randint(180, 255),0,0,0)
                        img_pil = Image.fromarray(seg)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(word_coords, f'{good_wording}', font=font, fill=word_color)
                        seg = np.array(img_pil)
                        word_display_count -= 1

                    seg[0:50, 540:640] = header_5
                    # seg[5:35, 5:35] = hp_header
                    # seg[40:70, 5:35] = cal_header
                    seg[3:69, 5:35] = hp_cal_header



                elif mode == "select":

                    if mute_dir == 0:
                        header_6 = overlayList[14] # music_on button activate 
                    else:
                        header_6 = overlayList[14] # mute button activate 

                    wording = f"Total Calories : {int(total_cal*0.01)}"
                    coords = (120, 10)
                    cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+407, coords[1]+52), (0, 0, 0), 2)
                    cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+405, coords[1]+50), (255, 255, 255), -1) 
                    img_pil = Image.fromarray(seg)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                    seg = np.array(img_pil)
                    

                    header_5 = overlayList[9]

                    # Checking for the click
                    #exit
                    if x2 < 100:
                        if y2<50:
                            
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
            
                                new_rows = pd.DataFrame([total_cal])
                                chart.add_rows(new_rows)
                                last_rows = new_rows

                                break

                        elif 90<=y2<190:
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
                        elif 290<=y2<390:
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

                    elif x1>540:
                        # jumping
                        if 90<=y1<190:
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
                        elif 290<=y1<390:
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
                    

                # FPS Counter logic
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if HP > 100:
                    HP = 100

                # if record:
                #     out.write(seg)

            else:
                wording = "Please Appear On The Screen"
                coords = (50, 220)
                cv2.rectangle(seg,(coords[0]-12, coords[1]-2), (coords[0]+507, coords[1]+52), (0, 0, 0), 2)
                cv2.rectangle(seg,(coords[0]-10, coords[1]), (coords[0]+505, coords[1]+50), (255, 255, 255), -1) 
                img_pil = Image.fromarray(seg)
                draw = ImageDraw.Draw(img_pil)
                draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                seg = np.array(img_pil)

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
            'λ°₯ νκ³΅κΈ°' : 310,
            'λ‘λ³Άμ΄' : 300,
            'μΌκ²Ήμ΄' : 460,
            'λΌλΌ' : 180,
            'νΌμ' : 404,
            'μΉν¨' : 249,
            'μ΄μ½λ°' : 240,
            'μ΄λ°₯' : 179,
            'μμΉκ΅­μ' : 447,
            'μμ΄μ€ν¬λ¦Ό' : 186,
            'νλ²κ±° μΈνΈ' : 956,
            'μ§μ₯λ©΄' : 785,
            'μ§¬λ½' : 464,
            'κ΅­λ°₯' : 470,
            'μλ©λ¦¬μΉ΄λΈ' : 4,
            'μ¬κ³Ό' : 57,
            'μ°μ ' : 65,
            'λ°λλ' : 93,
            'κ΅°κ³ κ΅¬λ§' : 124,
            'λ°©μΈν λ§ν ' : 2,
            'λλΆ' : 88,
            'μΆμ λ¬κ±' : 68,
            'μΈμ λ―Έ' : 220,
            'μΉμ¦ μΌμ΄ν¬' : 265,
            'λ‘€μΌμ΄ν¬' : 244,
            'μν¬λ¦ΌλΉ΅' : 220,
            'μλ²ν°' : 674,
            'λ°κ²νΈ' : 44,
            'μννΈμ½' : 145,
            'ν' : 136,
            'μ§μ₯λ©΄ + μ§¬λ½' : 1249,
            'μ§λ°©' : 2000,
            'μ λλ€! μ΄λ§νΌ' : 3000,
            'μ! μ λΆ λ€' : 999999999
        }

        new_food_calories = sorted(food_calories.items(), key=lambda x:x[1], reverse=False)

        st.empty()
        st.write("---")
        st.empty()

        for food, cal in new_food_calories:
            if total_cal == 0:
                st.title(f'π₯³ μ¨μ¬κΈ° μ΄λμ΄λΌλ νλ©΄ λμ£ ! πͺ')
                break

            elif total_cal < cal:
                st.title("π WoW~ μ΄λμ μ’ λΈμ¨κ΅°μ? π")
                st.title(f'π₯³ {food} κ²©ννμ΄μ! λΉ μ€! πͺ')
                break

        st.balloons()

        col_df_1, col_df_2 = st.columns(2)

        with col_df_1:
            st.write(counts_df.head())

        with col_df_2:
            st.pyplot(fig1)

###################################################################################################

@app.addapp(title='Hi Clicker')
def clicker():

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

    global score, x_enemy, y_enemy, count

    # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    # record = st.checkbox("Record Video")

    fontpath = "fonts/CarterOne-Regular.ttf"
    font = ImageFont.truetype(fontpath, 35)

    folderPath = "examples/Header_Clicker"
    myList = os.listdir(folderPath)

    # λ?μ΄μμ°λ μ΄λ―Έμ§ λ¦¬μ€νΈ
    overlayList = []

    # Header ν΄λμ imageλ₯Ό μλκ²½λ‘λ‘ μ§μ 
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    header_1 = overlayList[12]
    hp_cal_header = overlayList[14]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write('\n')
    with col2:
        level = st.slider("λ λ²¨μ μ ννμΈμ.", 1, 5, 1)
    with col3:
        st.write('')

    col4, col5, col6 = st.columns([1, 3, 1])
    with col4:
        st.write('\n')
    with col5:
        stframe = st.empty()
        yt_url = st.text_input('μ νλΈ [κ³΅μ  λ§ν¬] λ₯Ό λΆμ¬ λ£μ΄μ£ΌμΈμ.')
        st.text("λ°λͺ¨λ₯Ό μνμν€λ €λ©΄ 'demo'λ₯Ό μλ ₯νμΈμ.")
        st.text("μλ¦¬λ₯Ό λλ €λ©΄ 'MUTE' λ²νΌμ 2μ΄κ° Click! νμΈμ.")
    with col6:
        st.write('')

    # λΈνΈ λμ€λ μλ μ‘°μ  κ°μ΄ μ€μ΄λ€ μλ‘ μμ± μλκ° λΉ¨λΌμ§
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

    

    # if use_webcam:

    if yt_url:
        if yt_url == "demo":
            audio_file_path = "media/demo/audio.mp3"
            data_file_path = "media/demo/pitchdata.csv"
        else:
            # μ νλΈ λ§ν¬ μλ ₯μΌλ‘ λ€μ΄λ°μμ€λ λΆλΆ
            video_download_path = './media/video'
            audio_download_path = './media/audio'
            data_download_path = './media/data'

            if not os.path.exists(video_download_path):
                os.makedirs(video_download_path)

            video = YouTube(yt_url)
            video_type = video.streams.filter(progressive=True, file_extension="mp4").get_highest_resolution()
            video_type.download(video_download_path)

            # λ€μ΄λ°μ λΉλμ€μ μ΄λ¦μ video.mp4λ‘ λ°κΎΈκΈ° -> μ΄λ¬λ©΄ λΈλ νλλ§ μμ΄μΌνλ€λ λ¬Έμ μ .
            file_list = os.listdir(video_download_path)
            file_position = len(file_list)
            old_name = os.path.join(video_download_path, file_list[-1])
            new_name = os.path.join(video_download_path, f"{file_position}.mp4")

            os.rename(old_name, new_name)

            # mp4λ₯Ό mp3 λ‘ μ ν
            clip = mv.VideoFileClip(video_download_path + f"/{file_position}.mp4")
            clip.audio.write_audiofile(audio_download_path + f"/{file_position}.mp3")

            audio_file_path = audio_download_path + f"/{file_position}.mp3"

            Sound = parselmouth.Sound(audio_file_path)

            # 1μ΄λ¨μλ‘
            formant = Sound.to_formant_burg(time_step=1)
            # Pitchκ° μΆμΆ
            pitch = Sound.to_pitch()
            df = pd.DataFrame({"times": formant.ts()})

            times = []
            times.append(formant.ts())

            df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))

            data_file_path = data_download_path + f"/{file_position}.csv"
            df.to_csv(data_file_path)

        sound = pd.read_csv(data_file_path, index_col=0)

        df = sound['F0(pitch)']
        # sound['times'] = sound['times'].astype(int)
        normal_df = (df - df.min()) / (df.max() - df.min())
        pitch = normal_df.fillna(2)
        sound['F0(pitch)'] = pitch

        # λΈλ μ½μ
        sounds = {}  # λΉ λμλλ¦¬ μμ±
        pygame.mixer.init()
        sounds["slap"] = pygame.mixer.Sound("examples\Assets\Sounds\slap.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["slap"].set_volume(0.8)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["screaming"] = pygame.mixer.Sound("examples\Assets\Sounds\Effect.wav")  # μ¬μν  νμΌ μ€μ 
        sounds["screaming"].set_volume(0.6)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨
        sounds["song"] = pygame.mixer.Sound(audio_file_path)  # μ¬μν  νμΌ μ€μ 
        sounds["song"].set_volume(0.2)  # λ³Όλ₯¨ μ€μ  SOUNDS_VOLUME 0~1 μ¬μ΄μ κ°μ λ£μΌλ©΄ λ¨

        # class κ°μ²΄ μμ±
        movedetector = pm.poseDetector()

        cal = 0
        score = 0
        note_count = difficulty
        x_enemy = random.randint(50, 600)
        y_enemy = random.randint(50, 400)

        mp3_time = 0

        sounds["song"].play()

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
            # print(i_pitch)
            return result

        prevTime = 0

        ########################################################
        # mediapipe opencv logic
        ########################################################
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        while vid.isOpened():

            ret, img = vid.read()
            if not ret:
                continue

            image = img.copy()
            image = cv2.flip(image, 1)
            movedetector.findPose(image)
            footlmList = movedetector.findPosition(image, draw=False)

            if note_count == difficulty:
                mp3_time += 1

            # μν λΈνΈ μμ±
            if note_count > 0 and y_enemy < 220:
                note_count -= 1
                cv2.circle(image, (x_enemy, y_enemy), 25,
                           (0, 0, random.randint(0, 255)), 5)
            # μΌκ°ν λΈνΈ μμ±
            elif note_count > 0 and y_enemy >= 220 and y_enemy < 300:
                note_count -= 1
                pts = np.array([[x_enemy, y_enemy], [x_enemy + 60, y_enemy], [x_enemy + 30, y_enemy - 49]], np.int32)
                cv2.polylines(image, [pts], True, (0, 0, random.randint(0, 255)), 3)
            # μ¬κ°ν λΈνΈ μμ±
            elif note_count > 0 and y_enemy >= 300:
                note_count -= 1
                cv2.rectangle(image, (x_enemy, y_enemy), (x_enemy + 40, y_enemy + 40),
                              (0, 0, random.randint(0, 255)), 5)

            # μ¬μ©μκ° κ°μ§λ₯Ό λͺ» νμ λ
            if note_count == 0:
                x_enemy = random.randint(50, 600)
                y_enemy = make_pitch(sound, mp3_time)
                note_count = difficulty  # random.randint(30,40)

            if len(footlmList) != 0:

                # μμ μ’λ£
                x1, y1 = footlmList[20][1:3]
                x2, y2 = footlmList[19][1:3]

                # μμ΄ μ»€μ
                cv2.circle(image, (x1, y1), 5, (0, 0, 0), 2)
                cv2.circle(image, (x2, y2), 5, (0, 0, 0), 2)

                if (540 < x1 <= 640 and 0 < y1 <= 50) or (540 < x2 <= 640 and 0 < y2 <= 50):
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

                # λ΄ λͺΈμ μλͺ© κ·Έλ¦¬κΈ°
                cv2.circle(image, (footlmList[16][1:3]), 25, (0, 200, 0), 5)
                cv2.circle(image, (footlmList[15][1:3]), 25, (0, 200, 0), 5)

                # λ΄ λͺΈμ μΌκ°ν κ·Έλ¦¬κΈ°
                pts = np.array([[footlmList[25][1], footlmList[25][2]], [footlmList[25][1] + 60, footlmList[25][2]],
                                [footlmList[25][1] + 30, footlmList[25][2] - 49]], np.int32)
                cv2.polylines(image, [pts], True, (0, 200, 0), 3)

                pts = np.array([[footlmList[26][1], footlmList[26][2]], [footlmList[26][1] + 60, footlmList[26][2]],
                                [footlmList[26][1] + 30, footlmList[26][2] - 49]], np.int32)
                cv2.polylines(image, [pts], True, (0, 200, 0), 3)

                # λ¬ λͺΈμ μ¬κ°ν κ·Έλ¦¬κΈ°
                cv2.rectangle(image, (footlmList[31][1], footlmList[31][2]),
                              (footlmList[31][1] + 40, footlmList[31][2] + 40),
                              (0, 200, 0), 5)

                cv2.rectangle(image, (footlmList[32][1], footlmList[32][2]),
                              (footlmList[32][1] + 40, footlmList[32][2] + 40),
                              (0, 200, 0), 5)

                if y_enemy < 220:
                    # μ€λ₯Έμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # μΌμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.3
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.3
                # μΌκ°νμΌ λ
                elif y_enemy >= 220 and y_enemy < 310:
                    # μ€λ₯Έμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.4
                    # μΌμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.4
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.35
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                # μ¬κ°νμΌ λ
                elif y_enemy >= 310:
                    # μ€λ₯Έμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    if abs(footlmList[25][1] - x_enemy - 30) < 30 and abs(footlmList[25][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # μΌμͺ½ λ¬΄λ¦μΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[26][1] - x_enemy - 30) < 30 and abs(footlmList[26][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.4
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[31][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.35
                    # λ°λ‘ κ°μ§ νμ λ
                    elif abs(footlmList[32][1] - x_enemy - 20) < 30 and abs(footlmList[32][2] - y_enemy - 20) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("found")
                        sounds["slap"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score + 1
                        cal = cal + 0.35
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[15][1] - x_enemy) < 30 and abs(footlmList[15][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3
                    # μμΌλ‘ κ°μ§ νμ λ
                    elif abs(footlmList[16][1] - x_enemy) < 30 and abs(footlmList[16][2] - y_enemy) < 30:
                        x_enemy, y_enemy = 1000, 1000
                        print("fail")
                        sounds["screaming"].play()  # μμ μ¬μ
                        note_count = difficulty
                        mp3_time += 1
                        score = score - 1
                        cal = cal + 0.3

                image[0:50, 540:640] = header_1
                image[3:33, 5:35] = hp_cal_header

                
                if cal > 0:
                    cv2.rectangle(image, (40, 10), (40 + int(cal*5)*3, 28), (16, 117, np.random.randint(230,255)), cv2.FILLED)
                cv2.rectangle(image, (38, 8), (332, 30), (255, 255, 255), 2)


            else:
                wording = "Please Appear On The Screen"
                coords = (50, 220)
                cv2.rectangle(image,(coords[0]-12, coords[1]-2), (coords[0]+507, coords[1]+52), (0, 0, 0), 2)
                cv2.rectangle(image,(coords[0]-10, coords[1]), (coords[0]+505, coords[1]+50), (255, 255, 255), -1) 
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text(coords, f'{wording}', font=font, fill=(0,0,0,0))
                image = np.array(img_pil)

            # FPS Counter logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            # if record:
            #     out.write(image)

            image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            image = image_resize(image=image, width=900)
            stframe.image(image, channels='BGR', use_column_width='auto')


app.run()


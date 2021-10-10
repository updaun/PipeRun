from re import T
from mediapipe.python.solutions.face_mesh import FACE_CONNECTIONS
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
from settings import *
from game import Game, image_resize
from menu import Menu


#################################################################################

import hydralit as hy
import hydralit_components as hc

## 네비바 ##
app = hy.HydraApp(title='Simple Multi-Page App')

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
#################################################################################

class Cell:
    """A Cell can hold text, markdown, plots etc."""
    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f""".{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};}}"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()

    def plotly_chart(self, fig):
        self.inner_html = f"""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <body>
            <p>This should have been a plotly plot.
            But since *script* tags are removed when inserting MarkDown/ HTML i cannot get it to work</p>
            <div id='divPlotly'></div>
            <script>
                var plotly_data = {fig.to_json()}
                Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
            </script>
        </body>
        """

    def to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""

class Grid:
    """A (CSS) Grid"""
    def __init__(
        self, template_columns="1 1 1", gap="10px", background_color="#fff", color="#444"
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
        <style>
        
        .wrapper {{
        display: grid;
        grid-template-columns: {self.template_columns};
        grid-gap: {self.gap};
        background-color: {self.background_color};
        color: {self.color};
        }}
        .box {{
        background-color: {self.color};
        color: {self.background_color};
        border-radius: 5px;
        padding: 20px;
        font-size: 150%;
        }}
        table {{
            color: {self.color}
        }}
    </style>
    """

    def _get_cells_style(self):
        return (
            "<style>" + "\n".join([cell._to_style() for cell in self.cells]) + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell.to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell

    def select_block_container_style():
        """Add selection section for setting setting the max-width and padding
        of the main block container"""
        st.header("Block Container Style")
        max_width_100_percent = st.checkbox("Max-width: 100%?", False)
        if not max_width_100_percent:
            max_width = st.slider("Select max-width in px", 100, 2000, 1200, 100)
        else:
            max_width = 1200
        padding_top = st.number_input("Select padding top in rem", 0, 200, 5, 1)
        padding_right = st.number_input("Select padding right in rem", 0, 200, 1, 1)
        padding_left = st.number_input("Select padding left in rem", 0, 200, 1, 1)
        padding_bottom = st.number_input(
            "Select padding bottom in rem", 0, 200, 10, 1
        )
        _set_block_container_style(
            max_width,
            max_width_100_percent,
            padding_top,
            padding_right,
            padding_left,
            padding_bottom,
        )


    def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = False,
        padding_top: int = 5,
        padding_right: int = 1,
        padding_left: int = 1,
        padding_bottom: int = 10,
    ):
        if max_width_100_percent:
            max_width_str = f"max-width: 100%;"
        else:
            max_width_str = f"max-width: {max_width}px;"
        st.markdown(
                    f"""
            <style>
                .reportview-container .main .block-container{{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
            </style>
            """,
                    unsafe_allow_html=True,
                )


@st.cache
def get_dataframe() -> pd.DataFrame():
    """Dummy DataFrame"""
    data = [
        {"quantity": 1, "price": 2},
        {"quantity": 3, "price": 5},
        {"quantity": 4, "price": 8},
    ]
    return pd.DataFrame(data)

def get_plotly_fig():
    """Dummy Plotly Plot"""
    return px.line(
        data_frame=get_dataframe(),
        x="quantity",
        y="price"
    )


#################################################################################################

class App():
    global mp_drawing, mp_holistic, DEMO_IMAGE, DEMO_VIDEO, DEMO_VIDEO_RUNNING, interpreter, input_details, output_details, seq, action_seq, seq_length, actions, last_action

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    DEMO_IMAGE = './demo/demo.png'
    DEMO_VIDEO = './demo/demo.mp4'
    DEMO_VIDEO_RUNNING = './demo/running.mp4'

    @app.addapp(title='Home', is_home=True)
    def home():
        # googlefonts
        components.html(
            """
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Permanent+Marker&display=swap" rel="stylesheet">

            """)

        # bootstrap
        components.html(
            """
            <html>

            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@500&display=swap" rel="stylesheet">
           
            <style>
            .main_txt{font-size:40px; font-weight:bold;}
            .main_sm_txt{line-height: calc(1.4 + var(--space) / 100); font-size:22px; margin-top : 50px;}
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
                    <img src="https://www.pennmedicine.org/-/media/images/miscellaneous/fitness%20and%20sports/woman_exercise_home.ashx" class="d-block w-100" class="d-block w-100" alt="...">
                    <div class="carousel-caption d-none d-md-block">
                        <h5>Game with AI, Try it!</h5>
                        <p>Fun ways to get 30 minutes of physical activity today</p>
                    </div>
                    </div>
                    <div class="carousel-item">
                    <img src="https://media.cnn.com/api/v1/images/stellar/prod/210803081253-college-exercise-fitness-lead.jpg?q=w_1601,h_901,x_0,y_0,c_fill" class="d-block w-100" alt="...">
                    <div class="carousel-caption d-none d-md-block">
                        <h5>Let's have fun exercising together</h5>
                        <p>Anywhere, Anyone, with AI</p>
                    </div>
                    </div>
                    <div class="carousel-item">
                    <img src="https://www.wellandgood.com/wp-content/uploads/2020/03/GettyImages-1141568835.jpg" class="d-block w-100" alt="...">
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
                        <div class="col-md-6" >
                            <p class="main_txt">AI와 함께하는 헬스케어</p>
                            <p class="main_sm_txt">
                            인공신경망으로 정확한 운동 자세를 학습한 AI가 당신과 함께 합니다.<br>
                            카메라를 켜보세요. 당신의 모든 움직임을 인식하고 알려줍니다.<br>
                            이제 당신이 학습할 차레입니다!<br>
                            </p>

                        </div>
                        <div class="col-md-6">
                            <img src="https://images.pexels.com/photos/6707079/pexels-photo-6707079.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500">
                        </div>
                    </div>
                   
                
                   
                    <div class="row">
                        <div class="col-md-6">
                            <img src="https://images.pexels.com/photos/4498366/pexels-photo-4498366.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=210&w=500">  
                        </div>
                        <div class="col-md-6">
                            <p class="main_txt">리듬 타고 싶은 멜로디가 있나요?</p>
                            <p class="main_sm_txt">좋아하는 노래, TOP100, 지루한 인터넷 강의?<br>
                                소리만 있으면 무엇이든 가능합니다.<br>
                                일단 올려주세요!<br>
                                그리고 리듬에 몸을 맡기고 움직이면 됩니다.
                            </p>

                        </div>
                       
                    </div>
                     <div class="row" style="margin-top:100px; padding :100px 0 100px 0; ">
                        <div class="col-md-12 text-center">
                            <h2 class="">어렵기만 한 운동 자세, 이제 걱정마세요</h2>
                            <h5>카메라를 켜보세요. 당신의 모든 움직임을 인식하고 알려줍니다. </h5>
                        </div>
                    </div>
                </div>
            </body>
            <footer style="border-top:1px solid #222; margin-top:100px; padding:100px 50px 100px 50px;">
                <p>전다운</p>
                <p>Copyright © 1995-2021 Samsung. All Rights Reserved.</p>
            </footer>
            </html>
            """,
            height=2700)


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


    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="models/running_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.last_action
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    seq = []
    action_seq = []
    last_action = None
    seq_length = 30
    actions = ['run', 'stop']

############################################################################################################################################################


    @app.addapp(title='Try pygame')
    def try_pygame():
        global state

        st.markdown('In this Application we are using **Mediapipe** for creating a Holistic App. **Streamlit** is to create the Web Graphical User Interface (GUI)')
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                font-size:30px;
                background-color: rgb(204, 250, 250);
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
        # bootstrap
        # components.html(
        #     """
        #     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-uWxY/CJNBR+1zjPWmfnSnVxwRheevXITnMqoEIeG1LJrdI0GlVs/9cVSyPYXdcSF" crossorigin="anonymous">
        #     <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-kQtW33rZJAHjgefvhyyzcGF3C5TFyBQBA13V1RKPf4uH+bwyzQxZ6CmMZHmNBEfJ" crossorigin="anonymous"></script>
        #     <button type="button" class="btn btn-outline-primary">GAME START</button>""")

        
        if st.button("GAME START"):

            # Setup pygame/window --------------------------------------------- #
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100,32) # windows position
            pygame.init()
            pygame.display.set_caption(WINDOW_NAME)
            SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT),0,32)

            mainClock = pygame.time.Clock()

            # Fonts ----------------------------------------------------------- #
            fps_font = pygame.font.SysFont("coopbl", 22)

            # Music ----------------------------------------------------------- #
            pygame.mixer.music.load("Assets/Sounds/Komiku_-_12_-_Bicycle.mp3")
            pygame.mixer.music.set_volume(MUSIC_VOLUME)
            pygame.mixer.music.play(-1)

            # Variables ------------------------------------------------------- #
            state = "menu"

            # Creation -------------------------------------------------------- #
            game = Game(SCREEN)
            menu = Menu(SCREEN)



            # Functions ------------------------------------------------------ #
            def user_events():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        # sys.exit()
                        cv2.destroyAllWindows()
                        continue

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            # sys.exit()
                            cv2.destroyAllWindows()
                            continue

            def update():
                global state
                if state == "menu":
                    if menu.update() == "game":
                        game.reset() # reset the game to start a new game
                        state = "game"
                elif state == "game":
                    if game.update() == "menu":
                        state = "menu"
                pygame.display.update()
                mainClock.tick(FPS)



            # Loop ------------------------------------------------------------ #
            while True:

                # Buttons ----------------------------------------------------- #
                user_events()

                # Update ------------------------------------------------------ #
                update()

                # FPS
                if DRAW_FPS:
                    fps_label = fps_font.render(f"FPS: {int(mainClock.get_fps())}", 1, (255,200,20))
                    SCREEN.blit(fps_label, (5,5))

    ############################################################################################################################################################

    @app.addapp(title='Running Detection')
    def running_Detection():
        global last_action

        # Layout
        col1, col2 = st.columns(2)

        with col1:    
            use_webcam = st.button('Use Webcam')
            video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
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

        stframe = st.empty()

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

    @app.addapp(title='About App')
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

        st.set_option('deprecation.showfileUploaderEncoding', False)

        use_webcam = st.button('Use Webcam')
        record = st.checkbox("Record Video")

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

        # st.markdown("**Detected Faces, Hands and Pose**")
        # kpi1_text = st.markdown("0")
        


        max_faces = st.number_input('Maximum Number of Face', value=2, min_value=1)
        st.markdown('---')
        detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
        tracking_confidence = st.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
        st.markdown('---')

        st.markdown('## Output')

        stframe = st.empty()
        video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])

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


    # @app.addapp(title='Practice')
    def practice():

        def my_widget(key):
            st.subheader('Hello there!')    
            clicked = st.button("Click me " + key)
            # This works in the main area
        clicked = my_widget("first")
            # And within an expander
        my_expander = st.beta_expander("Expand", expanded=True)
        with my_expander:
            clicked = my_widget("second")
        # AND in st.sidebar!
        with st.sidebar:
            clicked = my_widget("third")

        st.markdown(
                """
        # Layout and Style Experiments

        The basic question is: Can we create a multi-column dashboard with plots, numbers and text using
        the [CSS Grid](https://gridbyexample.com/examples)?

        Can we do it with a nice api?
        Can have a dark theme?
        """
            )

        # My preliminary idea of an API for generating a grid
        with Grid("1 1 1") as grid:
            grid.cell(
                class_="a",
                grid_column_start=2,
                grid_column_end=3,
                grid_row_start=1,
                grid_row_end=2,
            ).markdown("# This is A Markdown Cell")
            grid.cell("b", 2, 3, 2, 3).text("The cell to the left is a dataframe")
            grid.cell("c", 3, 4, 2, 3).plotly_chart(get_plotly_fig())
            grid.cell("d", 1, 2, 1, 3).dataframe(get_dataframe())
            grid.cell("e", 3, 4, 1, 2).markdown("Try changing the **block container style** in the sidebar!")



app.run()
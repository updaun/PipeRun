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
from game import Game
from menu import Menu

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEMO_IMAGE = './demo/demo.png'
DEMO_VIDEO = './demo/demo.mp4'
DEMO_VIDEO_RUNNING = './demo/running.mp4'

st.title('Welcome to PipeRun')

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

st.sidebar.title('PipeRun Menu')
st.sidebar.subheader('parameters')

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


app_mode = st.sidebar.selectbox('Choose the App mode', ['real time plot test', 'try pygame','Running Detection', 'About App', 'Run on Image', 'Run on Video' ])

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/running_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

seq = []
action_seq = []
last_action = None
seq_length = 30
actions = ['run', 'stop']

# ############################################################################################################################################################


# if app_mode == 'basic page':

#     st.markdown(
#         """
#         <style>
#         [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
#             width:350px
#         }
#         [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
#             width:350px
#             margin-left: -350px
#         }
#         </style>
#         """,

#         unsafe_allow_html=True,
#     )

# ############################################################################################################################################################


############################################################################################################################################################


if app_mode == 'real time plot test':

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px;
            margin-left: -350px;
        }
        input {
        unicode-bidi:bidi-override;
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    # progress_bar = st.progress(0)
    # status_text = st.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)

    # for i in range(1, 101):
    #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows(new_rows)
    #     progress_bar.progress(i)
    #     last_rows = new_rows
    #     time.sleep(0.05)

    # progress_bar.empty()

    # # Streamlit widgets automatically run the script from top to bottom. Since
    # # this button is not connected to any other logic, it just causes a plain
    # # rerun.
    # st.button("Re-run")


    # with st.beta_expander("Upload MP3 Files"):
    #     uploaded_files = st.file_uploader("Select MP3 Files:", 
    #                                     accept_multiple_files=True,
    #                                     type='mp3')


    # input text box
    # user_input = st.text_input("5분 미만의 유튜브 공유 링크를 넣어주세요.")
    # st.write(user_input)

############################################################################################################################################################


elif app_mode == 'try pygame':

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

elif app_mode == 'Running Detection':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

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

    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])

    tffile = tempfile.NamedTemporaryFile(delete=False)

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
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
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)
    
    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    # kpi1, kpi2, kpi3 = st.columns(3)
    kpi1, kpi2, kpi3 = st.beta_columns(3)

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


elif app_mode == 'About App':
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

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

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

    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE 
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
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

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

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
    


    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])

    tffile = tempfile.NamedTemporaryFile(delete=False)

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
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
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)
    
    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    # kpi1, kpi2, kpi3 = st.columns(3)
    kpi1, kpi2, kpi3 = st.beta_columns(3)

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





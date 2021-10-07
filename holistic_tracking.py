import cv2
import mediapipe as mp
from settings import *
import numpy as np
import holistic_module as hm
import streamlit as st
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# detector = hm.HolisticDetector()

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



class HandTracking:
    def __init__(self):
        # self.hand_tracking = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_tracking = hm.HolisticDetector()
        self.hand_x = 0
        self.hand_y = 0
        self.results = None
        self.hand_closed = False
        self.stframe = st.empty()

    def scan_hands(self, image):
        rows, cols, _ = image.shape

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # image.flags.writeable = False
        # self.results = self.hand_tracking.process(image)
        
        # Draw the hand annotations on the image.
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self.hand_tracking.findHolistic(image, draw=True)
        h, w, c = image.shape
        self.face_lmList = self.hand_tracking.findFaceLandmark(image, draw=True)

        self.hand_closed = False

        if self.face_lmList:
            # for hand_landmarks in self.results.multi_hand_landmarks:
            x_, y_ = self.face_lmList[267][1], self.face_lmList[267][2]
            x, y = self.face_lmList[267][1] / w, self.face_lmList[267][2] / h
            # print(x,y)
            self.hand_x = int(x * SCREEN_WIDTH)
            self.hand_y = int(y * SCREEN_HEIGHT)

            x1, y1 = self.face_lmList[421][1] / w, self.face_lmList[421][2] / h

            cv2.circle(image, (x_,y_), 10, (0,200,0), cv2.FILLED)
            cv2.circle(image, (x_,y_), 15, (0,200,0), 2)

            if y1 > y:
                self.hand_closed = True


            # mp_drawing.draw_landmarks(
            #     image,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,)
                # mp_drawing_styles.get_default_hand_landmarks_style(),
                # mp_drawing_styles.get_default_hand_connections_style())
        return image

    def get_hand_center(self):
        return (self.hand_x, self.hand_y)


    def display_hand(self):
        # cv2.imshow("image", self.image)
        # cv2.waitKey(1)
        frame = cv2.resize(self.image, (0,0), fx=0.8, fy=0.8)
        frame = image_resize(image = frame, width = 640)
        self.stframe.image(frame, channels = 'BGR', use_column_width = True)
        

    def is_hand_closed(self):

        pass



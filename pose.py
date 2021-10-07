import cv2
import mediapipe as mp
from settings import *
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



class HandTracking:
    def __init__(self):
        self.hand_tracking = mp_pose.Pose(static_image_mode=True,
                                          model_complexity=1,
                                          smooth_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
        self.hand_x = 0
        self.hand_y = 0
        self.results = None
        self.hand_closed = False


    def scan_hands(self, image):
        # rows, cols, _ = image.shape

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        self.results = self.hand_tracking.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.pose_landmarks:
            for i in range(33):
                print(self.results.pose_landmarks.landmark[i].x, self.results.pose_landmarks.landmark[i].y)
                x, y = self.results.pose_landmarks.landmark[i].x, self.results.pose_landmarks.landmark[i].y

                self.hand_x = int(x * SCREEN_WIDTH)
                self.hand_y = int(y * SCREEN_HEIGHT)

                if x:
                    self.hand_closed = True
                    
            mp_drawing.draw_landmarks(
                            image,
                            self.results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            )
        return image

    def get_hand_center(self):
        return (self.hand_x, self.hand_y)


    def display_hand(self):
        cv2.imshow("image", self.image)
        cv2.waitKey(1)

    def is_hand_closed(self):

        pass


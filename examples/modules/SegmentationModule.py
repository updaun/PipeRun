import cv2
import mediapipe as mp
import numpy as np

class SegmentationFilter():
    def __init__(self,
               model_selection=1):
        self.model_selection = model_selection
        
        self.mpSegmentation = mp.solutions.selfie_segmentation
        self.segmentations = self.mpSegmentation.SelfieSegmentation(self.model_selection)


    def OneColor(self, img, color = (192, 192, 192)):
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = color
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        imgRGB.flags.writeable = False
        self.results = self.segmentations.process(imgRGB)
        imgRGB.flags.writeable = True
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        condition = np.stack(
            (self.results.segmentation_mask,) * 3, axis=-1) > 0.1

        img = np.where(condition, img, bg_image)

        return img

    def Image(self, img, img_path):
        bg_image = cv2.imread(img_path)
        bg_image = cv2.resize(bg_image, (640, 480))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        imgRGB.flags.writeable = False
        self.results = self.segmentations.process(imgRGB)
        imgRGB.flags.writeable = True
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        condition = np.stack(
            (self.results.segmentation_mask,) * 3, axis=-1) > 0.1

        img = np.where(condition, img, bg_image)

        return img
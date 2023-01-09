import cv2
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_vggface import utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class VideoCamera(object):
    def __init__(self, path_video='', conf=0.7):
        self.path_video = path_video
        self.conf = conf
        self.cur_frame = 0
        self.video = None
        self.lips = np.asarray([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]).reshape(1,-1)
        self.dict_face_area = {}
        self.dict_lips_area = {}

    def __del__(self):
        self.video.release()
        
    def preprocess_image(self, cur_fr):
        cur_fr = utils.preprocess_input(cur_fr, version=2)
        return cur_fr
        
    def channel_frame_normalization(self, cur_fr):
        cur_fr = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
        cur_fr = cv2.resize(cur_fr, (224,224), interpolation=cv2.INTER_AREA)
        cur_fr = img_to_array(cur_fr)
        cur_fr = self.preprocess_image(cur_fr)
        return cur_fr
        
    def norm_coordinates(self, 
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px
    
    def make_padding_lips(self, img, shape=(88,88), pad=False):
        if pad:
            if img.shape[0] > 0 and img.shape[1] > 0:
                factor_0 = shape[0] / img.shape[0]
                factor_1 = shape[1] / img.shape[1]
                factor = min(factor_0, factor_1)
                dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
                img = cv2.resize(img, dsize)
                diff_0 = shape[0] - img.shape[0]
                diff_1 = shape[1] - img.shape[1]
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'mean')
        if img.shape[0:2] != shape:
            img = cv2.resize(img, shape)   
        return tf.cast(img, tf.float16) / 255.

    def get_box(self, fl, w, h):
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)

            if landmark_px:
                idx_to_coors[idx] = landmark_px
   
        boxx_face = []
        # face
        if self.cur_frame in self.need_frames_for_pred_emotion:
            x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
            y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
            endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
            endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])

            (startX, startY) = (max(0, x_min), max(0, y_min))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            boxx_face = [startX, startY, endX, endY]
        
        # lips
        x_min_lips = np.min(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 0])
        y_min_lips = np.min(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 1])
        x_max_lips = np.max(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 0])
        y_max_lips = np.max(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 1])
        boxx_lips = [x_min_lips, y_min_lips, x_max_lips, y_max_lips]
        
        return boxx_face, boxx_lips
            
    def get_frame(self):
        self.video = cv2.VideoCapture(self.path_video)
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.need_frames_for_pred_emotion = list(range(1, total_frame+1, round(5*fps/25)))
        print('Name video: ', os.path.basename(self.path_video))
        print('Number total of frames: ', total_frame)
        print('FPS: ', fps)
        print('Video duration: {} s'.format(np.round(total_frame/fps, 2)))
        print('Frame width:', w)
        print('Frame height:', h)
        
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
            while self.video.isOpened():
                success, image = self.video.read()
                self.cur_frame += 1
                name_img = str(self.cur_frame).zfill(6)
                if image is None: break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        boxx_face, boxx_lips  = self.get_box(fl, w, h)
                        if self.cur_frame in self.need_frames_for_pred_emotion:
                            cur_fr_face = image[boxx_face[1]: boxx_face[3], boxx_face[0]: boxx_face[2]]
                            self.dict_face_area[name_img] = self.channel_frame_normalization(cur_fr_face)
                        cur_fr_lips = image[boxx_lips[1]: boxx_lips[3], boxx_lips[0]: boxx_lips[2]] 
                        self.dict_lips_area[name_img] = self.make_padding_lips(cur_fr_lips, pad=True)
        return self.dict_face_area, self.dict_lips_area, total_frame
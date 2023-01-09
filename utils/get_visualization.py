import cv2
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class VideoCamera(object):
    def __init__(self, path_video='', path_report='', path_save='', type_lips_model=0):
        self.path_video = path_video
        self.df = pd.read_csv(path_report)
        self.lips = np.asarray([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]).reshape(1,-1)
        self.prob = pd.DataFrame(self.df.drop(['frame', 'emo'], axis=1)).values
        self.pred_emo = self.df.emo[0]
        self.true_emo = self.path_video.split('_')[-2]
        self.true_phrase = self.path_video.split('_')[-3]
        self.prob_phrase = np.sum(self.prob, axis=0)/np.sum(self.prob)
        self.pred_phrase = np.argmax(self.prob_phrase)
        self.type_lips_model = type_lips_model
        self.label_emo = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
        self.label_phrase = ["DFA", "IEO", "IOM",
                             "ITH", "ITS", "IWL",
                             "IWW", "MTI", "TAI",
                             "TIE", "TSI", "WSI"]
        self.sort_pred = np.argsort(-self.prob)
        self.path_save = path_save
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cur_frame = 0
        self.video = None

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
                
        x_min = np.min(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 0])
        y_min = np.min(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 1])
        endX = np.max(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 0])
        endY = np.max(np.asarray(list(idx_to_coors.values()))[self.lips][:, :, 1])

        (startX, startY) = (max(0, x_min), max(0, y_min))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY)) 

        return startX, startY, endX, endY
    
    def get_metadata(self):
        overlay = self.fr.copy()
        fontScale = (self.fr.shape[0])/720
        
        if self.type_lips_model == 0:
            t_e = 'True emotion: ' + self.true_emo
            p_e = 'Pred emotion: ' + self.label_emo[self.pred_emo]
        elif self.type_lips_model == 1:
            if self.true_emo==0:
                t_e = 'True valence: ' + 'NEU'
            elif self.true_emo==1:
                t_e = 'True valence: ' + 'POS'
            else:
                t_e = 'True valence: ' + 'NEG'
            
            if self.pred_emo==0:
                p_e = 'Pred valence: ' + 'NEU'
            elif self.pred_emo==1:
                p_e = 'Pred valence: ' + 'POS'
            else:
                p_e = 'Pred valence: ' + 'NEG'
        elif self.type_lips_model == 2:
            if self.pred_emo==0:
                p_e = 'Pred binary: ' + 'Neutral'
            else:
                p_e = 'Pred binary: ' + 'Emotional'
                
            if self.true_emo==0:
                t_e = 'True binary: ' + 'Neutral'
            else:
                t_e = 'True binary: ' + 'Emotional'

        t_p = 'True phrase: ' + self.true_phrase
        p_p = 'Pred phrase: ' + self.label_phrase[self.pred_phrase] + ' {:.2f}%'.format(self.prob_phrase[self.pred_phrase]*100)
        
        if self.type_lips_model == 0:
            t_m = 'Type model: ' + '6-Emotions'
        elif self.type_lips_model == 1:
            t_m = 'Type model: ' + 'Valence'
        elif self.type_lips_model == 2:
            t_m = 'Type model: ' + 'Binary'
            
        thickness = 1
        text_color = (255, 255, 255)
        text_width, text_height = cv2.getTextSize(t_e, self.font, fontScale, thickness)
        text_width = text_width[0]+int((self.fr.shape[0]*6)/720)

        start_text_1 = self.fr.shape[0]-text_height
        start_text_2 = self.fr.shape[0]-text_height*4
        start_text_3 = self.fr.shape[0]-text_height*7
        start_text_4 = self.fr.shape[0]-text_height*10
        start_text_5 = self.fr.shape[0]-text_height*13
        opacity = 0.3
        
        cv2.addWeighted(overlay, opacity, self.fr, 1 - opacity, 0, self.fr)
        
        cv2.rectangle(overlay, (int((self.fr.shape[0]*2)/720), start_text_1-int((self.fr.shape[0]*20)/720)), (int((self.fr.shape[0]*2)/720)+text_width, self.fr.shape[0]), (255, 0, 255), -1)
        cv2.putText(self.fr, t_e, (int((self.fr.shape[0]*2)/720), start_text_1), self.font, fontScale, text_color, thickness)
        
        cv2.rectangle(overlay, (int((self.fr.shape[0]*2)/720), start_text_2-int((self.fr.shape[0]*20)/720)), (int((self.fr.shape[0]*2)/720)+text_width, self.fr.shape[0]), (255, 0, 255), -1)
        cv2.putText(self.fr, p_e, (int((self.fr.shape[0]*2)/720), start_text_2), self.font, fontScale, text_color, thickness)
        
        cv2.rectangle(overlay, (int((self.fr.shape[0]*2)/720), start_text_3-int((self.fr.shape[0]*20)/720)), (int((self.fr.shape[0]*2)/720)+text_width, self.fr.shape[0]), (255, 0, 255), -1)
        cv2.putText(self.fr, t_p, (int((self.fr.shape[0]*2)/720), start_text_3), self.font, fontScale, text_color, thickness)
        
        cv2.rectangle(overlay, (int((self.fr.shape[0]*2)/720), start_text_4-int((self.fr.shape[0]*20)/720)), (int((self.fr.shape[0]*2)/720)+text_width, self.fr.shape[0]), (255, 0, 255), -1)
        cv2.putText(self.fr, p_p, (int((self.fr.shape[0]*2)/720), start_text_4), self.font, fontScale, text_color, thickness)
        
        cv2.rectangle(overlay, (int((self.fr.shape[0]*2)/720), start_text_5-int((self.fr.shape[0]*20)/720)), (int((self.fr.shape[0]*2)/720)+text_width, self.fr.shape[0]), (255, 0, 255), -1)
        cv2.putText(self.fr, t_m, (int((self.fr.shape[0]*2)/720), start_text_5), self.font, fontScale, text_color, thickness)

        # cv2.rectangle(fr, (2, start_text), (2+text_width, fr.shape[0]), (255, 0, 0), 2)

#         return fr

    def draw_prob(self, emotion_yhat, best_n, startX, startY, endX, endY):
    
        label = '{}: {:.2f}%'.format(self.label_phrase[best_n[0]], emotion_yhat[best_n[0]]*100)
        
        lw = max(round(sum(self.fr.shape) / 2 * 0.003), 2)
        
        text2_color = (255, 0, 255)
        p1, p2 = (startX, startY), (endX, endY)
        cv2.rectangle(self.fr, p1, p2, text2_color, thickness=lw, lineType=cv2.LINE_AA)
                
        tf = max(lw - 1, 1)
        fontScale = 2
        text_fond = (0,0,0)
        text_width_2, text_height_2 = cv2.getTextSize(label, self.font, lw / 3, tf)
        text_width_2 = text_width_2[0]+round(((p2[0]-p1[0])*10)/360)
        center_face = p1[0]+round((p2[0]-p1[0])/2)
        
        cv2.putText(self.fr, label, (center_face-round(text_width_2/2), p1[1] - round(((p2[0]-p1[0])*20)/360)), 
                    self.font, lw / 3, text_fond, thickness=tf, lineType=cv2.LINE_AA)
        
        cv2.putText(self.fr, label, (center_face-round(text_width_2/2), p1[1] - round(((p2[0]-p1[0])*20)/360)), 
                    self.font, lw / 3, text2_color, thickness=tf, lineType=cv2.LINE_AA)
            
    def get_video(self):
        self.video = cv2.VideoCapture(self.path_video)
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.path_save += '.mp4'
        vid_writer = cv2.VideoWriter(self.path_save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
            while self.video.isOpened():
                success, image = self.video.read()
                self.fr = image
                name_img = str(self.cur_frame).zfill(6)
                if image is None: break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        startX, startY, endX, endY  = self.get_box(fl, w, h)
                        prob = self.prob[self.cur_frame]
                        best = self.sort_pred[self.cur_frame]
                        self.draw_prob(prob, best, startX, startY, endX, endY)
                        self.get_metadata()
                self.cur_frame += 1

                vid_writer.write(self.fr)
            vid_writer.release()
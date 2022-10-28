import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from utils import sequences
from utils import get_face_areas_2
from tqdm import tqdm
from utils.get_models import load_weights_EE, load_weights_LSTM

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='D:/Databases/CREMA-D/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report_iemocap/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default="C:/Users/Elena Ryumina/Desktop/work/CMU-MOSEI/models/EmoAffectnet/EmoAffectnet.h5",
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default="C:/Users/Elena Ryumina/Desktop/work/CMU-MOSEI/models/LSTM/IEMOCAP.h5",
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

def pred_one_video(path):
    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    detect = get_face_areas_2.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frame()
    name_frames = list(dict_face_areas.keys())
    face_areas = dict_face_areas.values()
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    features = EE_model(np.stack(face_areas))
    seq_paths, seq_features = sequences.sequences(name_frames, features)
    pred = LSTM_model(np.stack(seq_features)).numpy()
    all_pred = []
    all_path = []
    for id, c_p in enumerate(seq_paths):
        c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1])+1)]
        c_pr = [pred[id]]*len(c_f)
        all_pred.extend(c_pr)
        all_path.extend(c_f)    
    m_f = [str(i).zfill(6) for i in range(int(all_path[-1])+1, total_frame+1)] 
    m_p = [all_pred[-1]]*len(m_f)
    
    df=pd.DataFrame(data=all_pred+m_p, columns=label_model)
    df['frame'] = all_path+m_f
    df = df[['frame']+ label_model]
    df = sequences.df_group(df, label_model)
    
    if not os.path.exists(args.path_save):
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    df.to_csv(os.path.join(args.path_save,filename), index=False)
    end_time = time.time() - start_time
    mode = stats.mode(np.argmax(pred, axis=1))[0][0]
    print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_model[mode])
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in tqdm(enumerate(path_all_videos)):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    pred_all_video()
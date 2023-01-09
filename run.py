import argparse
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from scipy import stats
from utils import sequence_modeling
from utils import get_face_areas
from tqdm import tqdm
from utils.get_models import load_weights_EE, load_weights_LSTM
from utils import three_d_resnet_bi_lstm 

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='./test_video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report_iemocap/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default="./models/EMOAffectNet_LSTM/EmoAffectnet/weights.h5",
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default="./models/EMOAffectNet_LSTM/LSTM/weights.h5",
                    help='Path to a model for emotion prediction')
parser.add_argument('--type_lips_model', type=int, default=0,
                    help='0: 6 Emotion, 1: VALENCE, 2: BINARY')
                  

args = parser.parse_args()

def pred_one_video(path):
    start_time = time.time()
    label_emo = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
    label_phrase = ["DFA", "IEO", "IOM",
                    "ITH", "ITS", "IWL",
                    "IWW", "MTI", "TAI",
                    "TIE", "TSI", "WSI"]
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, dict_lips_areas, total_frame = detect.get_frame()
    name_frames = list(dict_face_areas.keys())
    face_areas = dict_face_areas.values()
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    features = EE_model(np.stack(face_areas)).numpy()
    seq_paths, seq_features = sequence_modeling.get_sequence(name_frames, features)
    pred_emo = LSTM_model(np.stack(seq_features)).numpy()
    pred_sum = np.sum(pred_emo, axis=0)
    sum = np.sum(pred_sum)
    pred_emo = pred_sum/sum
    sort_pred = np.argsort(-pred_emo)
    if sort_pred[0]==3:
        pred_emo=sort_pred[1]
    else:
        pred_emo=sort_pred[0]
        
    model_LR = three_d_resnet_bi_lstm.build_three_d_resnet_18((60, 88, 88, 3), 12, 'softmax', None,True, '3D')
    model_LR.build((None, 60, 88, 88, 3))
    
    if args.type_lips_model == 0:
        weights = './models/6 EMOTIONS/{}/weights.h5'.format(label_emo[pred_emo])
        model_LR.load_weights(weights)
    elif args.type_lips_model == 1:
        if pred_emo==0:
            weights = './models/VALENCE/{}/weights.h5'.format('NEU')
        elif pred_emo==1:
            weights = './models/VALENCE/{}/weights.h5'.format('POS')
        else:
            weights = './models/VALENCE/{}/weights.h5'.format('NEG')
        model_LR.load_weights(weights)
    else:
        if pred_emo==0:
            weights = './models/BINARY/{}/weights.h5'.format('NEU')
        else:
            weights = './models/BINARY/{}/weights.h5'.format('not_NEU')
        model_LR.load_weights(weights)
        
    name_frames_lips = list(dict_lips_areas.keys())    
    seq_paths_lips, seq_features_lips = sequence_modeling.get_sequence(name_frames_lips, np.stack(dict_lips_areas.values()), win = 60, step = 30)
    pred_phrase = model_LR(np.stack(seq_features_lips)).numpy()    
    
    all_pred = []
    all_path = []
    for id, c_p in enumerate(seq_paths_lips):
        c_pr = [pred_phrase[id]]*len(c_p)
        all_pred.extend(c_pr)
        all_path.extend(seq_paths_lips[id])    
    df=pd.DataFrame(data=all_pred, columns=label_phrase)
    df['frame'] = all_path
    df = df[['frame']+ label_phrase]
    df = sequence_modeling.df_group(df, label_phrase)
    df['emo'] = len(df)* [pred_emo]
    
    if not os.path.exists(args.path_save):
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    df.to_csv(os.path.join(args.path_save,filename), index=False)
    end_time = time.time() - start_time
    mode = stats.mode(np.argmax(pred_phrase, axis=1))[0][0]
    print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_phrase[mode])
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in tqdm(enumerate(path_all_videos)):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    pred_all_video()
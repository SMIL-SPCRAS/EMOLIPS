import pandas as pd
from tqdm import tqdm
from decimal import *

def new_list(l, count = 50):
    len_a = len(l)
    if count >= len_a:
        return l
    c = len_a / count
    res = []
    prev = 1
    cnt = 0
    for i in l:
        if prev >= len_a:
            break
        cnt += 1
        dec = int(Decimal(prev).to_integral_value(rounding = ROUND_HALF_UP))
        if cnt == dec:
            prev += c
            res.append(i)
    return res

def frame_thinning(df, fps, c_fr):
    c_n_frames = round(len(df))/(c_fr*fps/30)
    n_frames= new_list(range(len(df)), count = c_n_frames)
    return n_frames

def get_data(path, id_phrases):
    id_sp = [int(i.split('\\')[1].split('_')[0]) for i in path]
    name_video = [i.split('\\')[1] for i in path]
    phrases = [i.split('\\')[1].split('_')[1] for i in path]
    emotions = [i.split('\\')[1].split('_')[2] for i in path]
    id_class = [id_phrases[i.split('\\')[1].split('_')[1]] for i in path]
    df = pd.DataFrame(columns=['path_images', 'name_video', 'emotion', 'id_class','phrase', 'id_speaker'])
    df['path_image'] = path
    df['name_video'] = name_video
    df['emotion'] = emotions
    df['id_class'] = id_class
    df['phrase'] = phrases
    df['id_speaker'] = id_sp
    df = df[df.name_video!='1064_IEO_DIS_MD']
    df = df.reset_index(drop=True)
    return df


def get_sequence(all_path, all_feature, win = 10, step = 5):
    seq_path = []
    seq_feature = []
    for id_cur in range(0, len(all_path)+1, step):
        need_id = id_cur+win
        curr_path = all_path[id_cur:need_id]
        curr_FE = all_feature[id_cur:need_id].tolist()
        if len(curr_path) < win and len(curr_path) != 0:
            curr_path.extend([curr_path[-1]]*(win - len(curr_path)))
            curr_FE.extend([curr_FE[-1]]*(win - len(curr_FE)))
        if len(curr_path) != 0:
            seq_path.append(curr_path)
            seq_feature.append(curr_FE)
    return seq_path, seq_feature


def df_group(df, label_model):
    df_group = df.groupby(['frame']).agg({i:'mean' for i in label_model})

    df_group.reset_index(inplace=True)
    df_group = df_group.sort_values(by=['frame'])
    df_group.reset_index(drop=True)
    return df_group
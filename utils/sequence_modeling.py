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

def get_sequence(df, len_seq=10, step = 2, fps=30, c_fr=False):
    uniq_n = list(df.name_video.unique())
    paths = []
    names = []
    phrases = []
    labels = []
    for i in tqdm(uniq_n):
        c_df = df[df.name_video==i]
        c_df = c_df.reset_index(drop=True)
        if c_fr: 
            n_fr = frame_thinning(c_df, fps, c_fr)
            c_df = c_df[c_df.index.isin(n_fr)]
            c_df = c_df.reset_index(drop=True)
        for c_id in range(0, len(c_df), round(len_seq/step)):
            n_id = c_id+len_seq
            c_path = c_df.loc[c_id:n_id-1].path_image.tolist()
            c_name = c_df.loc[c_id:n_id-1].name_video.tolist()
            c_phrase = c_df.loc[c_id:n_id-1].phrase.tolist()
            c_labels = c_df.loc[c_id:n_id-1].id_class.tolist()
            if len(c_path) < len_seq and len(c_path) != 0:
                c_path.extend([c_path[-1]]*(len_seq - len(c_path)))
                c_phrase.extend([c_phrase[-1]]*(len_seq - len(c_phrase)))
                c_name.extend([c_name[-1]]*(len_seq - len(c_name)))
                c_labels.extend([c_labels[-1]]*(len_seq - len(c_labels)))
            if len(c_path) != 0:
                paths.append(c_path)
                phrases.append(c_phrase)
                names.append(c_name)
                labels.append(c_labels)            
    return paths, names, phrases, labels



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
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def get_group(age, age_group):
    for i in range(len(age_group)):
        min = int(age_group[i].split('-')[0])
        max = int(age_group[i].split('-')[1])
        if age >= min and age < max:
            return age_group[i]
        
def get_id_sp(path):
    phrases = {"IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting",
                "IWW": "I wonder what this is about", "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
                "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment", "DFA": "Don't forget a jacket",
                "ITS": "I think I've seen this before","TSI": "The surface is slick", "WSI": "We'll stop in a couple of minutes"}

    phrases = dict(sorted(phrases.items()))

    id_phrases = {list(phrases.keys())[i]: i for i in range(len(phrases.keys()))}

    df_md = pd.read_csv(path)

    age_group = ['20-30', '30-40', '40-50', '50-80']

    df_md['age_group'] = [get_group(i, age_group) for i in df_md.Age]

    seed = 0

    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.30,random_state=seed)
    for train_ind, t_v_ind in sss.split(df_md, df_md[['age_group', 'Sex']]):
        id_sp_train = df_md[df_md.index.isin(train_ind)].ActorID.values.tolist()
        df_md_valid_test = df_md[df_md.index.isin(t_v_ind)]
        df_md_valid_test = df_md_valid_test.reset_index(drop=True)

    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.32,random_state=seed)
    for test_ind, val_in in sss.split(df_md_valid_test, df_md_valid_test[['age_group', 'Sex']]):
        id_sp_test = df_md_valid_test[df_md_valid_test.index.isin(test_ind)].ActorID.values.tolist()
        id_sp_valid = df_md_valid_test[df_md_valid_test.index.isin(val_in)].ActorID.values.tolist()
    return id_phrases, id_sp_train, id_sp_valid, id_sp_test
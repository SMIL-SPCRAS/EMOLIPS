from sklearn.metrics import recall_score
from  math import ceil
from tensorflow.keras.callbacks import *
from tqdm import tqdm

import datetime
import csv
import os
import pandas as pd
import numpy as np

class   Metrics(Callback):

    def __init__(self, path_save_weight = 'models/', verbose=0, patience=0, n_cl=12, name_app = ''):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.patience = patience
        self.n_cl = n_cl
        self.path_save_model = path_save_weight
        self.checkpoints = os.path.join(self.path_save_model, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+'_{}'.format(name_app)
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
            df = pd.DataFrame(columns=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])
            df.to_csv(self.checkpoints + '/valid.csv', index=False) 
            df.to_csv(self.checkpoints + '/test.csv', index=False) 
        print(self.checkpoints)
            
            
    def on_train_begin(self, logs={}):
        self.val_recalls = []
        self.val_recall = 0
        self.stop_flag_recall = 0
        self.num_epochs = 0

    def pred_prob(self, validation_generator, batch, seq):
        validation_generator.reset()
        step=ceil(validation_generator.n/batch)
        prob_all = []
        for i in tqdm(range(step)):
            image_curr = validation_generator.next()
            image_curr = image_curr.reshape(-1, seq, 224, 224, 3)
            prob_all.append(self.model.predict(image_curr))

        prob_all_2 = list(itertools.chain(*prob_all))
        prob = np.asarray(prob_all_2)
        print(prob.shape)
        return prob

    def get_prey_truey_all_video(self, name_x_new, truey, predy, n_class):
            emotion_count = np.zeros((n_class))
            list_true = []
            list_pred = []
            name = None
            name_list = []
            for i in range(len(name_x_new)):
                if name == None:
                    name = name_x_new[i]
                if name_x_new[i] == name:
                    true = truey[i]
                    if type(predy[i]) == int:
                        emotion_count[predy[i]] += 1
                    else:
                        emotion_count += predy[i]
                else:
                    list_true.append(true) 
                    list_pred.append(emotion_count/np.sum(emotion_count))
                    name = name_x_new[i]
                    emotion_count = np.zeros((n_class))
                    true = truey[i]
                    if type(predy[i]) == int:
                        emotion_count[predy[i]] += 1
                    else:
                        emotion_count += predy[i] 
                if i == len(name_x_new)-1:
                    list_true.append(true) 
                    list_pred.append(emotion_count/np.sum(emotion_count))
                if name not in name_list:
                    name_list.append(name)

            list_pred= np.asarray(list_pred)
            pred_max = np.argmax(list_pred, axis = 1).tolist()

            return name_list, list_true, pred_max
        
    def get_pred(self, generator):
        pred_y_val = []
        true_y_val = []
        paths_val = []
        for x, y, p in tqdm(generator):
            paths_val.extend(p)
            pred_y_val.extend(self.model(x).numpy())
            true_y_val.extend(np.argmax(y, axis = 1))
        name_list, val_targ, val_predict = self.get_prey_truey_all_video(paths_val, true_y_val, pred_y_val, self.n_cl)
        return name_list, val_targ, val_predict
    
    def get_dataframe(self, name, true, pred, subset):
        df = pd.DataFrame(columns=['name', 'phrase', 'emo', 'intensity', 'true', 'pred',])
        df['name'] = [i for i in name]
        df['phrase'] = [i.split('_')[1] for i in name]
        df['emo'] = [i.split('_')[2] for i in name]
        df['pred'] = pred
        df['true'] = true
        df['intensity'] = [i.split('_')[3] for i in name]
        df = df.reset_index(drop=True)
        scores = []
        for em in df['emo'].unique():
            pred_c = df[df.emo == em].pred.values
            true_c = df[df.emo == em].true.values
            score = recall_score(true_c, pred_c, average='macro', zero_division=0)
#             print(em, score)
            scores.append(score)
        filename = self.checkpoints + '/report_{}_{}.csv'.format(subset, self.num_epochs)
        df.to_csv(filename, index=False)
        with open(self.checkpoints + '/{}.csv'.format(subset), 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(scores)
            
    def on_epoch_end(self, epoch, logs={}):

        self.num_epochs += 1
        val_name_list, val_targ, val_predict = self.get_pred(self.model.valid_generator)
        self.get_dataframe(val_name_list, val_targ, val_predict, 'valid')
        score = recall_score(val_targ, val_predict, average='macro', zero_division=0)
        self.val_recalls.append(score)
        logs['val_recall'] = score
        
        test_name_list, test_targ, test_predict = self.get_pred(self.model.test_generator)
        self.get_dataframe(test_name_list, test_targ, test_predict, 'test')

        if score >= self.val_recall:
            self.stop_flag_recall = 0
            self.model.save_weights(self.checkpoints + '/weights_{}_{}.h5'.format(epoch, '_'.join(str('{0:.4%}'.format(score)).split('.'))))
            if self.verbose > 0:
                print('Recall improved from {} to {}'.format(self.val_recall, score))
                self.val_recall = score
        else:
            self.stop_flag_recall += 1
            if self.verbose > 0:
                print('Recall did not improve.')

        if self.stop_flag_recall > self.patience:
            if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))
            self.model.stop_training = True
           
        print('max_val_recall {}\n'.format(max(self.val_recalls)))
        print('current_val_recall {}\n'.format(score))
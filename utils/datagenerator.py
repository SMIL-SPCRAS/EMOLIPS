import tensorflow as tf
from skimage.io import imread
import cv2
import numpy as np

class DataGeneratorTrain(tf.keras.utils.Sequence):
    def __init__(self, ps, ls, bs=3, n_c=12, shape=(112, 112), chan=3, p_a=0.40, shuffle=True, pad=False):
        self.ps = ps # list of paths
        self.ls = ls # list of labels
        self.bs = bs # bach size
        self.n_c = n_c # number of classes
        self.shape = shape # image size
        self.chan = chan # number of image channels
        self.p_a = int(len(self.ps)*p_a) # maximum probability for image augmentation by the MixUp method
        self.c_a = 0 # counter to count the number of augmented image sequences
        self.shuffle = shuffle # subset shuffling
        self.pad = pad # padding the image by average values in height to the desired size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ps) / self.bs))

    def __getitem__(self, ind):
        ind = self.ind[ind*self.bs:(ind+1)*self.bs]
        c_ps = [self.ps[k] for k in ind]
        c_ls = [self.ls[k] for k in ind]
        x, y = self.__data_generation(c_ps, c_ls)
        return x, y
    
    def make_padding(self, img):
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.shape[0] / img.shape[0]
            factor_1 = self.shape[1] / img.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            diff_0 = self.shape[0] - img.shape[0]
            diff_1 = self.shape[1] - img.shape[1]
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'mean')
        if img.shape[0:2] != self.shape:
            img = cv2.resize(img, self.shape)   
        return img
    
    def read_img(self, path):
        x = imread(path)
        if self.pad:
            x = self.make_padding(x)
        else:
            x = tf.image.resize(x, self.shape)
        if self.chan == 1:
            x = tf.image.rgb_to_grayscale(x)
        return x
    
    def load_data(self, ps, ls):
        xs = []
        augment = False
        if self.p_a != 0 and self.c_a < self.p_a:
            augment = random.choice([False, True])
        if augment:
            random_s = random.choice(self.ind)
            l = float(np.random.beta(0.5, 0.5, 1))
            y = np.zeros(self.n_c)
            y[ls[0]] = 1
            for id, p_c in enumerate(ps):
                x = self.read_img(p_c)
                x2 = self.read_img(self.ps[random_s][id])
                x = tf.cast(x,dtype=tf.float16) * l + tf.cast(x2,dtype=tf.float16) * (1 - l)
                xs.append(x)
            y2 = np.zeros(self.n_c)
            y2[self.ls[random_s][0]] = 1
            y = y * l + y2 * (1 - l)
            self.c_a += 1
            return xs, y
        else:
            y = np.zeros(self.n_c)
            y[ls[0]] = 1
            for id, p_c in enumerate(ps):
                x = self.read_img(p_c)
                xs.append(tf.cast(x,dtype=tf.float16))
            return xs, y

    def on_epoch_end(self):
        self.ind = np.arange(len(self.ps))
        self.c_a = 0
        if self.shuffle == True:
            np.random.shuffle(self.ind)

    def __data_generation(self, c_ps, c_ls):
        x = []
        y = []
        for ps, ls in zip(c_ps, c_ls):
            c_x, c_y = self.load_data(ps, ls)
            x.append(c_x)
            y.append(c_y)
        return tf.cast(x, tf.float16) / 255., tf.cast(y, tf.float16)
    
class DataGeneratorTest(tf.keras.utils.Sequence):
    def __init__(self, ps, ls, ns, bs=3, n_c=12, shape=(112, 112), chan=3, shuffle=False, pad=False):
        self.ps = ps # list of paths
        self.ls = ls # list of labels
        self.ns = ns # list of names 
        self.bs = bs # bach size
        self.n_c = n_c # number of classes
        self.shape = shape # self.chan = chan # number of image channels
        self.chan = chan # number of image channels
        self.shuffle = shuffle # subset shuffling
        self.pad = pad # padding the image by average values in height to the desired size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ps) / self.bs))

    def __getitem__(self, ind):
        ind = self.ind[ind*self.bs:(ind+1)*self.bs]
        c_ps = [self.ps[k] for k in ind]
        c_ls = [self.ls[k] for k in ind]
        c_ns = [self.ns[k] for k in ind]
        x, y, n = self.__data_generation(c_ps, c_ls, c_ns)
        return x, y, n
    
    def make_padding(self, img):
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.shape[0] / img.shape[0]
            factor_1 = self.shape[1] / img.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            diff_0 = self.shape[0] - img.shape[0]
            diff_1 = self.shape[1] - img.shape[1]
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'mean')
        if img.shape[0:2] != self.shape:
            img = cv2.resize(img, self.shape)   
        return img
    
    def read_img(self, path):
        x = imread(path)
        if self.pad:
            x = self.make_padding(x)
        else:
            x = tf.image.resize(x, self.shape)
        if self.chan == 1:
            x = tf.image.rgb_to_grayscale(x)
        return x
    
    def load_data(self, ps, ls, ns):
        y = np.zeros(self.n_c)
        y[ls[0]] = 1
        xs = []
        for p_c in ps:
            x = self.read_img(p_c)
            xs.append(x)
        return xs, y, ns[0]

    def on_epoch_end(self):
        self.ind = np.arange(len(self.ps))
        if self.shuffle == True:
            np.random.shuffle(self.ind)

    def __data_generation(self, c_ps, c_ls, c_ns):
        x = []
        y = []
        n = []
        for ps, ls, ns in zip(c_ps, c_ls, c_ns):
            c_x, c_y, c_n = self.load_data(ps, ls, ns)
            x.append(c_x)
            y.append(c_y)
            n.append(c_n)
        return tf.cast(x, tf.float16) / 255., tf.cast(y, tf.float16), n
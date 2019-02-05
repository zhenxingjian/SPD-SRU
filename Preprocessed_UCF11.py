import os
import numpy as np
import pickle
import datetime
import pdb
import cv2

from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

GLOBAL_MAX_LEN = 1492

def get_clips(class_name):
    files = os.listdir(data_path + class_name)
    files.sort()
    clip_list = []
    for this_file in files:
        if '.DS_Store' not in this_file and 'Annotation' not in this_file:
            clips = os.listdir(data_path + class_name + '/' + this_file)
            clips.sort()
            for this_clip in clips:
                if '.DS_Store' not in this_clip and 'Annotation' not in this_file:
                    clip_list.append( data_path + class_name + '/' + this_file + '/' + this_clip )
    return clip_list


def load_data(inds, mode = 'train'):
    N = len(inds)
    X = np.zeros((N, GLOBAL_MAX_LEN, 120*160*3), dtype='int8')
    if mode == 'train':
        set = train_set
    else:
        set = test_set
    for i in range(N):
        read_in = open(set[0][inds[i]])
        this_clip = pickle.load(read_in)[0] # of shape (nb_frames, 240, 320, 3)
        read_in.close()
        # flatten the dimensions 1, 2 and 3
        this_clip = this_clip.reshape(this_clip.shape[0], -1) # of shape (nb_frames, 240*320*3)
        this_clip = (this_clip - 128.).astype('int8')   # this_clip.mean()
        X[i] = pad_sequences([this_clip], maxlen=GLOBAL_MAX_LEN, truncating='post', dtype='int8')[0]
    Y = set[1][inds]
    return [X, Y]


# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)

# Settings:

CV_setting = 0  # [0, 1, 2, 3, 4]
model_type = 1  # 0 for GRU, 1 for LSTM
use_TT = 0      # 0 for non-TT, 1 for TT

# Had to remove due to anonymity
data_path = './UCF11_updated_mpg/'
write_out_path = ''

classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
           'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']

clips = [None]*11
labels = [None]*11
sizes = np.zeros(11)
for k in range(11):
    this_clip = get_clips(classes[k])
    clips[k] = this_clip
    sizes[k] = len(this_clip)
    labels[k] = np.repeat([k], sizes[k])

# flatten both lists
clips = np.array( [item for sublist in clips for item in sublist] )
labels = np.array( [item for sublist in labels for item in sublist] )
labels = to_categorical(labels)

shuffle_inds = np.random.choice(range(len(clips)), len(clips), False)
totalclips = clips[shuffle_inds]
totallabels = labels[shuffle_inds]
if not os.path.isdir("/processed_data/"):
    os.mkdir("processed_data")
# pdb.set_trace()
# iterate through all clips and store the length of each:
for xj in range(40):
    clips = totalclips[xj*40:xj*40+40]
    labels = totallabels[xj*40:xj*40+40]
    # pdb.set_trace()
    data = []
    length_of_frames = []
    lengths = np.zeros(len(clips))
    for l in range(len(clips)):
        print clips[l]
        cap = cv2.VideoCapture(clips[l])
        ret = True
        tempdata = []
        count = 0
        while(ret):
            ret, frame = cap.read()
            if ret:
                count = count + 1
                cv2.imshow('frame',frame)
                frame = cv2.resize(frame,(160,120))
                tempdata.append(frame)
        length_of_frames.append(count)

        # for i in range(count,GLOBAL_MAX_LEN):
        #     tempdata.append(tempdata[count-1])


        tempdata = np.asarray(tempdata,dtype=np.float32) / 255.
        data.append(tempdata)
    length_of_frames = np.asarray(length_of_frames)   
    data = np.asarray(data)
    np.save('processed_data/data'+str(xj)+'.npy',data)
    np.save('processed_data/label'+str(xj)+'.npy',labels)
    np.save('processed_data/length'+str(xj)+'.npy',length_of_frames)
    # pdb.set_trace()

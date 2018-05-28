import os
import numpy as np
import pickle
import datetime
import pdb
import cv2

from keras.preprocessing.sequence import pad_sequences



def read_data(idx,datapath,matrix_length,k):
    '''
    k is every k samples take one.
    for example, a = [1,2,3,4,5,6,7]; k = 2; result is [1,3,5,7]
    '''
    datas = np.load(datapath+'data'+str(idx)+'.npy')
    labels = np.load(datapath+'label'+str(idx)+'.npy')
    lengths = np.load(datapath+'length'+str(idx)+'.npy')
    data = []
    for i in range(len(datas)):
        datas[i] = datas[i][range(0,lengths[i],k),...]
        data.append(  pad_sequences([datas[i]], maxlen=matrix_length, truncating='post', dtype='float32')[0] )
    data = np.asarray(data)
    return data,labels


if __name__ == '__main__':
    GLOBAL_MAX_LEN = 1492
    data,label = read_data(1,'path',GLOBAL_MAX_LEN,2)
    pdb.set_trace()


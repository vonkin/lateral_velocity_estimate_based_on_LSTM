import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


class LSTMPredictor():
    def __init__(self,data_folder):
        forward_steps = 5
        freq = 200
        self.cache_folder = './cache/'
        self.data_folder = '../data/' + data_folder + '/'   #test_data_5 70km/h; _6 30km/h ; _7 50km/h
        #self.model = keras.models.load_model('../program_LSTM/temp_model/test_LSTM_with_vx_1031.h5')
        self.model = keras.models.load_model('../program_LSTM/temp_model/test_LSTM_with_vx_good.h5')
        #self.model = keras.models.load_model('../program_LSTM/temp_model/test_dnn_with_vx.h5')
        self.forward_steps = forward_steps
        self.ratio = int(2000/freq)
        self.X_names = ['Ax',
                        'Ay',
                        'engineSpeed',
                        'gearStat',
                        'steeringAngle',
                        'throttle',
                        'wheelSpeed_L1',
                        'wheelSpeed_L2',
                        'wheelSpeed_R1',
                        'wheelSpeed_R2',
                        'Vx',
                        'yawRate']
        self.Y_names = ['Vy'] 
        
            
    def find_cache(self):
        if self.data_folder == '../data/test_data_5/':
            self.cache_file = self.cache_folder + 'LSTM_70' + self.file_name + '.npz'
        elif self.data_folder == '../data/test_data_6/':
            self.cache_file = self.cache_folder + 'LSTM_30' + self.file_name + '.npz'
        elif self.data_folder == '../data/test_data_7/':
            self.cache_file = self.cache_folder + 'LSTM_50' + self.file_name + '.npz'
        elif self.data_folder == '../data/test_data_8/':
            self.cache_file = self.cache_folder + 'LSTM_110' + self.file_name + '.npz'
            
        #print(self.cache_file)
        if os.path.exists(self.cache_file):
            #print("exists!")
            A = np.load(self.cache_file)
            self.Y_ = A['Y_']
            self.Y_pred = A['Y_pred']
            return True
        else:
            #print('not found!')
            return False

    def load_file(self):
        data = sio.loadmat(self.data_file)
        X = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.X_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.X_names)
        Y = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.Y_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.Y_names)
        return X,Y
    
    def step_seperate(self,X,Y):
        X_ = np.array([[[X[var][i+ii] for var in self.X_names] for ii in range(self.forward_steps)] for i in range(int(len(X)-(self.forward_steps-1)))])
        Y_ = np.array([[Y[var][i + (self.forward_steps-1)] for var in self.Y_names] for i in range(int(len(Y)-(self.forward_steps-1)))])
        return X_,Y_    

    def predict(self,file_name):
        self.file_name = file_name
        self.data_file = self.data_folder + self.file_name
        if self.find_cache():
            return self.Y_,self.Y_pred
        else:
            X,Y = self.load_file()
            X_,Y_ = self.step_seperate(X,Y)
            Y_pred = self.model.predict(X_)
            self.Y_ = Y_
            self.Y_pred = Y_pred
            np.savez(self.cache_file,Y_=self.Y_,Y_pred=self.Y_pred)
            return self.Y_,self.Y_pred



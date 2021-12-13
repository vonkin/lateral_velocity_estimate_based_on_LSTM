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


class Tester():
    def __init__(self,model,freq,forward_steps,test_data_folder='1'):
        self.model = model
        self.forward_steps = forward_steps
        self.ratio = int(2000/freq)
        self.test_data_folder_list = ['test_data_{num}/'.format(num=char) for char in test_data_folder]
        self.data_folder = '../data/'
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

    def load_file(self,file_name):
        data = sio.loadmat(file_name)
        X = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.X_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.X_names)
        Y = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.Y_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.Y_names)
        return X,Y
    
    def step_seperate(self,X,Y):
        X_ = np.array([[[X[var][i+ii] for var in self.X_names] for ii in range(self.forward_steps)] for i in range(int(len(X)-(self.forward_steps-1)))])
        Y_ = np.array([[Y[var][i + (self.forward_steps-1)] for var in self.Y_names] for i in range(int(len(Y)-(self.forward_steps-1)))])
        return X_,Y_

    def draw_figure(self,Y_,Y_pred):
        for var in range(Y_.shape[1]):
            plt.figure(figsize=(20,5))
            plt.plot(np.arange(Y_.shape[0]),Y_[:,var])
            plt.plot(np.arange(Y_pred.shape[0]),Y_pred[:,var])
            plt.show()        

    def predict_file(self,file_name):
        X,Y = self.load_file(file_name)
        X_,Y_ = self.step_seperate(X,Y)
        Y_pred = self.model.predict(X_)
        self.draw_figure(Y_,Y_pred)   

    def test(self):
        for folder in self.test_data_folder_list:
            f_list = os.listdir(self.data_folder + folder)
            for file in f_list:
                if file.endswith(".mat"):
                    print(self.data_folder + folder + file)
                    self.predict_file(self.data_folder + folder + file)       


class DataLoader():
  
    def __init__(self,train_data_folder='12',test_data_folder='1',forward_steps=5,freq=200,pace=1,valid_size=0.1):
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
        self.ratio = int(2000/freq)
        self.forward_steps = forward_steps
        self.pace = pace
        self.valid_size = valid_size
        self.data_folder = '../data/'
        self.train_data_folder_list = ['train_data_{num}/'.format(num=char) for char in train_data_folder]
        self.test_data_folder_list = ['test_data_{num}/'.format(num=char) for char in test_data_folder]
        self.cache_folder = './cache_data/'
        self.cache_name = '{freq}_{train_data_folder}_{test_data_folder}_{steps}_{pace}_{valid_size}.npz'.format(train_data_folder=train_data_folder,
                                                                                                                 test_data_folder=test_data_folder,
                                                                                                                 steps="%03d" % forward_steps,
                                                                                                                 freq=freq,
                                                                                                                 valid_size=str(valid_size),
                                                                                                                 pace=pace)
        self.cache_file = self.cache_folder + self.cache_name
        self.X_all = None
        self.Y_all = None
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        
    def find_cache(self):
        print(self.cache_file)
        if os.path.exists(self.cache_file):
            print("exists!")
            A = np.load(self.cache_file)
            self.X_train = A['X_train']
            self.Y_train = A['Y_train']
            self.X_valid = A['X_valid']
            self.Y_valid = A['Y_valid']
            self.X_test = A['X_test']
            self.Y_test = A['Y_test']
            return True
        else:
            print('not found!')
            return False

    def load_file(self,file_name):
        data = sio.loadmat(file_name)
        X = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.X_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.X_names)
        Y = pd.DataFrame(data=[[data[var][i*self.ratio][0] for var in self.Y_names] for i in range(int(data['steeringAngle'].size/self.ratio))],
                        columns = self.Y_names)
        return X,Y

    def step_seperate(self,X,Y):
        #X_ = np.array([[[X[var][i+ii] for var in self.X_names] for ii in range(self.forward_steps)] for i in range(int(len(X)-(self.forward_steps-1)))])
        #Y_ = np.array([[Y[var][i + (self.forward_steps-1)] for var in self.Y_names] for i in range(int(len(Y)-(self.forward_steps-1)))])
        #xx = [X[var][0] for var in self.X_names]
        #print(xx)
        X_ = np.array([[[X[var][i*self.pace + ii] for var in self.X_names] for ii in range(self.forward_steps)] for i in range(int((len(X)-self.forward_steps)/self.pace))])
        Y_ = np.array([[Y[var][i*self.pace + self.forward_steps - 1] for var in self.Y_names] for i in range(int((len(Y)-self.forward_steps)/self.pace))])
        return X_,Y_

    def load_train_set(self):
        self.X_all = np.empty([0,self.forward_steps,len(self.X_names)])
        self.Y_all = np.empty([0,len(self.Y_names)])
        for folder in self.train_data_folder_list:
            f_list = os.listdir(self.data_folder + folder)
            for file in f_list:
                if file.endswith(".mat"):
                    print(self.data_folder + folder + file)
                    X,Y = self.load_file(self.data_folder + folder + file)
                    X_,Y_ = self.step_seperate(X,Y)
                    self.X_all = np.concatenate((self.X_all,X_))
                    self.Y_all = np.concatenate((self.Y_all,Y_))

    def load_test_set(self):
        self.X_test = np.empty([0,self.forward_steps,len(self.X_names)])
        self.Y_test = np.empty([0,len(self.Y_names)])
        for folder in self.test_data_folder_list:
            f_list = os.listdir(self.data_folder + folder)
            for file in f_list:
                if file.endswith(".mat"):
                    print(self.data_folder + folder + file)
                    X,Y = self.load_file(self.data_folder + folder + file)
                    X_,Y_ = self.step_seperate(X,Y)
                    self.X_test = np.concatenate((self.X_test,X_))
                    self.Y_test = np.concatenate((self.Y_test,Y_))

    def load(self):
        if self.find_cache():
            return self.X_train,self.Y_train,self.X_valid,self.Y_valid,self.X_test,self.Y_test
        else:
            self.load_train_set()
            self.load_test_set()
            self.X_train,self.X_valid,self.Y_train,self.Y_valid = train_test_split(self.X_all,self.Y_all,test_size=self.valid_size,random_state = 2021)
            np.savez(self.cache_file,
                     X_train=self.X_train,
                     Y_train=self.Y_train,
                     X_valid=self.X_valid,
                     Y_valid=self.Y_valid,
                     X_test=self.X_test,
                     Y_test=self.Y_test)
            return self.X_train,self.Y_train,self.X_valid,self.Y_valid,self.X_test,self.Y_test



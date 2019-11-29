import os
import numpy as np
import pickle
class Kalman_filter(object):
    def __init__(self, 
                 order=4,
                 time=0.01
                ):
        self.order = order
        self.time = time
        self.F = self.transitionMatrix(time)
        self.X = self.stateMatrix()
        self.B = self.controlMatrix(time)
        self.U = self.controlVector()
        self.P = self.stateVarMatrix()
        self.R = self.measureNoiseCovMatrix(time)
        self.Q = self.processNoiseCovMatrix(time)
        self.H = self.measurementMatrix()
        self.I = self.identityMatrix()
        
        self.FT= self.T(self.F)
        self.HT= self.T(self.H)
        
    # kalman_filter matrix setup
    def transitionMatrix(self, time=0):
        return np.array([[1, 0, time, 0],[0, 1, 0, time],[0, 0, 1, 0],[0, 0, 0, 1],])
    def stateMatrix(self):
        return np.zeros((self.order,1))
    def controlMatrix(self, time=0):
        t_ = (time**2)/2
        return np.array([[t_, 0],[0, t_],[time, 0],[0, time]])
    def controlVector(self, ax=0, ay=0):
        return np.array([[ax], [ay]])
    def stateVarMatrix(self):
        return np.eye(self.order) * 5000
    def measureNoiseCovMatrix(self, time=0):
        R = np.eye(int(self.order/2))
        return R * time if time != 0 else R
    def processNoiseCovMatrix(self, time=0):
        Q = np.eye(self.order)
        return Q * time if time != 0 else Q
    def measurementMatrix(self):
        return np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
    def identityMatrix(self):
        return np.eye(self.order)
    
    # calculation of array
    def dot(self, p, *args):
        for arg in args:
            p = np.dot(p, arg)
        return p
    def T(self, array):
        return array.transpose()
    def inverse(self, array):
        return np.linalg.inv(array)
    
    #setter, getter
    def set_state(self, state):
        self.X = state
    def set_varMatrix(self, covMatrix):
        self.P = covMatrix
    def get_state(self):
        return self.X
    def get_varMatrix(self):
        return self.P
    
    # kalman process:
    def predict(self):
        self.X = self.get_state()
        self.P = self.get_varMatrix()
        self.X = self.dot(self.F, self.X) + self.dot(self.B, self.U)
        self.P = self.dot(self.F, self.P, self.FT) + self.Q
        self.set_state(self.X)
        self.set_varMatrix(self.P)
        return self.X, self.P
    
    def update(self, measurement):
        self.X = self.get_state()
        self.P = self.get_varMatrix()
        A = self.dot(self.P, self.HT)
        B = self.inverse(self.dot(self.H, self.P, self.HT)+self.R)
        kalmanGain = self.dot(A, B)
        S = measurement.reshape((2,1)) - self.dot(self.H, self.X)
        self.X += self.dot(kalmanGain, S)
        self.P = self.dot((self.I - self.dot(kalmanGain, self.H)), self.P)
        self.set_state(self.X)
        self.set_varMatrix(self.P)
        return self.X, self.P

    def get_measurement(self):
        CURRENT_PATH = os.path.join(os.getcwd(), "lib/")
        DATA = []
        for i in os.listdir(CURRENT_PATH):
            if "data" in i: DATA.append(i)
        data1, data2, data3, data4 = [], [], [], []
        for data in DATA:
            path = os.path.join(CURRENT_PATH, data)
            for i in os.listdir(path):
                pick = os.path.join(path, i)
                if str(i[0]) == "1":
                    data1.append(pick)
                elif str(i[0]) == "2":
                    data2.append(pick)
                elif str(i[0]) == "3":
                    data3.append(pick)
                elif str(i[0]) == "4":
                    data4.append(pick)


        data1 = sorted(data1)
        data2 = sorted(data2)
        data3 = sorted(data3)
        data4 = sorted(data4)

        data1[2], data1[0] = data1[0], data1[2]
        data1[3], data1[2] = data1[2], data1[3]

        data2[2], data2[0] = data2[0], data2[2]
        data2[3], data2[2] = data2[2], data2[3]

        data3[2], data3[0] = data3[0], data3[2]
        data3[3], data3[2] = data3[2], data3[3]

        data4[2], data4[0] = data4[0], data4[2]
        data4[3], data4[2] = data4[2], data4[3]

        return data1, data2, data3, data4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def dot(p, *args):
    for arg in args:
        p = np.dot(p, arg)
    return p

def F(t):
    return np.array([[1, 0, t, 0],
                     [0, 1, 0, t],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],])

def B(t):
    t_ = (t**2)/2
    return np.array([[t_, 0],
                     [0, t_],
                     [t, 0],
                     [0, t]])

def U(ax, ay):
    return np.array([[ax],
                     [ay]])

def inverse(array):
    return np.linalg.inv(array)

true = pickle.load(open("b_velocity/true.pickle","rb"))
track1 = pickle.load(open("b_velocity/track1.pickle","rb"))
track2 = pickle.load(open("b_velocity/track2.pickle","rb"))
track3 = pickle.load(open("b_velocity/track3.pickle","rb"))

# for observation position
ZZ = [track1, track2, track3]
j = 1
for Z in ZZ:
    # for F and B
    t = 0.01
    
    Z[:,0] += np.random.normal(0, 2, 200)
    Z[:,1] += np.random.normal(0, 4, 200)

    FT = F(t).transpose()

    # for state X
    x_data = true[:,0]
    y_data = true[:,1]

    X = np.zeros((4,1))
    X_= np.zeros((4,1))

    # for variance P
    P = np.eye(4)
    
    # for observation error
    R = np.eye(2)
    if t != 0:
	    R *= t
    
    # for noise
    W = 1e-2

    #processNoiseCov
    Q = np.eye(4)
    if t != 0:
	    Q *= t
    
    # for Identity Matrix
    I = np.eye(4)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    HT= H.transpose()

    kalmanX, kalmanY = [], []
    var_err_X, var_err_Y = [], []
    error_obv, error_kal = [], []
    
    sx = 0
    sy = 0
    for i in range(len(x_data)):
        sx += (x_data.mean() - x_data[i])**2
        sy += (y_data.mean() - y_data[i])**2
        
    x_var = sx/len(x_data)
    y_var = sy/len(y_data)
    #==================================================================
    # formula
    for i in range(len(true)):
        ax = float(np.random.normal(0, np.sqrt(x_var), 1))
        ay = float(np.random.normal(0, np.sqrt(y_var), 1))

        X_= dot(F(t), X) + dot(B(t), U(ax, ay)) + W

        P_= dot(F(t), P, FT) + Q
        
        K = dot(dot(P_, HT), inverse(dot(H, P_, HT) + R))

        S = Z[i].reshape((2,1)) - dot(H, X_)

        X = X_ + dot(K, S)

        P = dot((I - dot(K, H)), P_)

        error_obv.append( (true[i,0]-Z[i,0])**2+(true[i,1]-Z[i,1])**2)
        error_kal.append( (true[i,0]-X[0,0])**2+(true[i,1]-X[1,0])**2)

        kalmanX.append(X[0,0])
        kalmanY.append(X[1,0])

        var_err_X.append(P[0,0])
        var_err_Y.append(P[1,1])

    plt.subplot(3,3,j)
    print("mean_square:")
    print((sum(error_kal)/len(error_kal)) / (sum(error_obv)/len(error_obv)))
    plt.plot(true[:,0],true[:,1])
    plt.plot(Z[:,0],Z[:,1],c='orange')
    plt.plot(kalmanX, kalmanY, c='g')

    plt.subplot(3,3,3+j)
    plt.plot(range(1,len(error_obv)*100,100), error_obv, c='orange')
    plt.plot(range(1,len(error_kal)*100,100), error_kal, c='g')

    plt.subplot(3,3,6+j)
    plt.plot(range(len(var_err_X)), var_err_X, c='orange')
    plt.plot(range(len(var_err_Y)), var_err_Y, c='g')
    j += 1

plt.show()
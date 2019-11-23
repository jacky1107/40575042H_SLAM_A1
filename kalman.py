import numpy as np
import matplotlib.pyplot as plt

size = 100

x = np.linspace(0, 100, size)

# Z = np.random.normal(0,2, size) + x - np.random.normal(3, 5, size)
Z = 1 + np.random.normal(0,2, size)

X = np.array([-10,-100]) #狀態

P = np.array([[1, 0], #狀態斜方差矩陣
              [0, 1]])

F = np.array([[1, 1], #狀態轉移矩陣
              [0, 1]])#每秒鐘採樣

Q = np.array([[1e-4, 0],#狀態轉移斜方差矩陣
              [0, 1e-4]])

H = np.array([1,0]) #觀測矩陣

R = 1 #觀測噪聲方差

FT = np.transpose(F)
HT = np.transpose(H)

I  = np.eye(2)

plt.ion()
plt.plot(x, np.ones(size))
# plt.plot(x, Z)
sca = plt.scatter( -10, -100 ); plt.pause(0.01)

for i in range(size):
    
    X_ = F*X
    P_ = F*P*FT + Q
    K  = P_*HT*np.linalg.inv(H*P_*HT + R)
    X  = X_ + K*(Z[i] - H*X_)
    P  = (I - K*H) * P_

    # if 'sca' in globals(): sca.remove()    
    sca = plt.scatter( x[i], X[0][0] ); plt.pause(0.01)


    

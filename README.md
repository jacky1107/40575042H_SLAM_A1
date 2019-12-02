# Kalman Filter

author: Jacky Wang

All code is written in Python, if there is any question please feel free to ask.

# How to Compile the program

Please make sure you have installed those packages.
```
numpy
matplotlib
pickle
```

- troubleshooting: pickle is not available if python.version < 3.6

Running:
``
python main.py [-s] [save_img, default:True]
``
The saved image will save into output folder  
After running main.py, it will show these infomation below:  
1. The true path of the robot  
2. The output of the Kalman Filter for the path of the robot  
3. In a separate subwindow show the variance of the Kalman Filter against time  
4. In a separate subwindow show the error of the Kalman Filter path against time  
5. Average error  
6. variance of error  

# Object detection example using Kalman filter
Interesting features and extensions of Kalman filter.
After running the code, make sure you have installed this package.
```
pip install opencv-python
```
Running:
``
python find_targeting_kalman_filter.py
``

This example will detect object with color blue.  
And green point is predicted value of Kalman value.  
Red point is true point which is centroid of object.

The result shows that the green point will track the red point as much as possible.  

# Kalman filter notes

F: transition Matrix  
X: state matrix  
U: control variable matrix  
W: noise in the process  
t: time for 1 circle  
H: measurement Matrix  

P: state covariance matrix (error in the estimate)  
If P -> 0:  
Then measurement updates are mostly ignored

P = [[ var_11, cvar12],
     [ cvar12, var_22]]  
If the estimate error for the one variable position is completely independent of the other variable velocity
Then the covariance elements = cvar12 = 0

Q: process noise covariance matrix  # processNoiseCov  
(keep the P from becoming too small or going to zero)

R: measurement covariance matrix (error in the measurement) #measurementCovMatrix  
K: kalman gain   
(weight factor based on comparing the error in the estimate to the error in the measurement)

if R -> 0:  
	then K -> 1 (adjust primarily with the measurement update)

if R -> unlimited:  
	then K -> 0 (adjust primarily with the predicted state)

# Reference
http://www.sherrytowers.com/kalman_filter_method.pdf  
http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf  
http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf  
https://zhuanlan.zhihu.com/p/40413223  
https://dsp.stackexchange.com/questions/50026/using-the-kalman-filter-given-acceleration-to-estimate-position-and-velocity  
https://www.youtube.com/watch?v=CaCcOwJPytQ&list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT  
# kalman_filter

author: Jacky Wang

All code is written in Python, if there is any problem please feel free to write commit below.

# How to Compile the program

Please make sure you have install those packages.
```
numpy
matplotlib
pickle
```

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

Q: process noise covariance matrix  ### processNoiseCov
(keep the P from becoming too small or going to zero)

R: measurement covariance matrix (error in the measurement) ### measurementCovMatrix
K: kalman gain 
(weight factor based on comparing the error in the estimate to the error in the measurement)

if R -> 0:
    then K -> 1 (adjust primarily with the measurement update)

if R -> unlimited:
    then K -> 0 (adjust primarily with the predicted state)

# Reference
http://www.sherrytowers.com/kalman_filter_method.pdf \n
http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
https://zhuanlan.zhihu.com/p/40413223
https://dsp.stackexchange.com/questions/50026/using-the-kalman-filter-given-acceleration-to-estimate-position-and-velocity
https://www.youtube.com/watch?v=CaCcOwJPytQ&list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT



(c) any interesting features and extensions of your assignment.

    

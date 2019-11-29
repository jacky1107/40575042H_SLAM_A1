# kalman_filter

author: Jacky Wang
# http://www.sherrytowers.com/kalman_filter_method.pdf

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

Q: process noise covariance matrix  #processNoiseCov
(keep the P from becoming too small or going to zero)

R: measurement covariance matrix (error in the measurement) #measurementCovMatrix
K: kalman gain 
(weight factor based on comparing the error in the estimate to the error in the measurement)

if R -> 0:
    then K -> 1 (adjust primarily with the measurement update)

if R -> unlimited:
    then K -> 0 (adjust primarily with the predicted state)




(a) what other sources you used apart from the lecture material used in class during your work on the assignment:
    

(b) how to compile and run your program:


(c) any interesting features and extensions of your assignment.

    

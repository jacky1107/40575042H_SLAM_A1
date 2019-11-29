import numpy as np
import matplotlib.pyplot as plt
import pickle
from lib.kalman_filter import Kalman_filter

kf = Kalman_filter()
datas = kf.get_measurement()

name = ["position", "velocity", "acceleration"]
count = 1
for data in datas:
    plt.figure()
    for measure in data:
        Z = pickle.load(open(measure,"rb"))
        if "true" in measure: 
            true = Z
            index = 0
            continue

        t = 0.01
        if name[0] in measure:
            t = 0
            draw = 1
        elif name[1] in measure: 
            draw = 2

        kf = Kalman_filter(4, time=t)

        kalmanX, kalmanY = [], []
        error_obv, error_kal = [], []
        var_kal = []
        for i in range(len(Z)):
            if name[2] in measure:
                ax = float(np.random.normal(0, 3, 1))
                ay = float(np.random.normal(0, 3, 1))
                kf.U = kf.controlVector(ax, ay)
                draw = 3
            
            x_predict, p_predict = kf.predict()
            X, P = kf.update(Z[i])

            error_obv.append( (true[i,0]-Z[i,0])**2+(true[i,1]-Z[i,1])**2)
            error_kal.append( (true[i,0]-X[0,0])**2+(true[i,1]-X[1,0])**2)

            kalmanX.append(X[0,0])
            kalmanY.append(X[1,0])

            var_kal.append(P[0,0])

        print(f"data{count}:", name[index])
        print(f'time is {t}')
        plt.subplot(3,3,draw)
        plt.title(name[index])
        if index == 0: plt.ylabel("path")
        mse = round((sum(error_kal)/len(error_kal)) / (sum(error_obv)/len(error_obv)), 2)
        print(f"mean square error is {mse}")
        plt.plot(true[:,0],true[:,1], label="true", c='r')
        plt.scatter(Z[:,0],Z[:,1],c='orange', s=2, label="measure")
        plt.plot(kalmanX, kalmanY, c='g', label="kalman")
        plt.legend()

        plt.subplot(3,3,3+draw)
        if index == 0: plt.ylabel("error")
        plt.plot(range(len(error_obv)), error_obv, c='orange', label="measure")
        plt.plot(range(len(error_kal)), error_kal, c='g', label="kalman")
        plt.legend()
        var_kal = np.array(var_kal) * (1 / max(var_kal))
        stop = min(var_kal)
        for i, v in enumerate(var_kal):
            if v == stop:
                print(f'variance reduced to {round(stop,2)} after {i} iteration')
                break
            
        plt.subplot(3,3,6+draw)
        if index == 0: plt.ylabel("variance")
        plt.xlabel(f"error={mse}")
        plt.plot(range(len(var_kal)), var_kal, c='g', label="kalman")
        plt.legend()
        print("")
        index += 1

    file_name = f'output/data{count}.png'
    plt.savefig(file_name)
    count += 1
    
plt.show()


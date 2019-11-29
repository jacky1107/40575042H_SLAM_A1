import numpy as np
import matplotlib.pyplot as plt
import pickle
from lib.kalman_filter import Kalman_filter

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", 
    "--save",
    dest="save",
    help="optional argument",
    default="True")
args = parser.parse_args()

save = args.save.lower()

kf = Kalman_filter()
datas = kf.get_measurement()

name = ["position", "velocity", "acceleration"]
count = 1

for data in datas:
    plt.figure(figsize=(20,7))
    mse = []
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
        error_o, error_k = [], []
        var_kal = []
        last_vx, last_vy = 0, 0
        for i in range(len(Z)):
            x_predict, p_predict = kf.predict()
            X, P = kf.update(Z[i])

            if name[2] in measure:
            	ax = float((last_vx-X[2,0])/t)
            	ay = float((last_vy-X[3,0])/t)
            	kf.U = kf.controlVector(ax, ay)
            	draw = 3

            error_o.append( (true[i,0]-Z[i,0])**2+(true[i,1]-Z[i,1])**2)
            error_k.append( (true[i,0]-X[0,0])**2+(true[i,1]-X[1,0])**2)

            error_obv.append( (true[i,0]-Z[i,0])+(true[i,1]-Z[i,1]))
            error_kal.append( (true[i,0]-X[0,0])+(true[i,1]-X[1,0]))

            kalmanX.append(X[0,0])
            kalmanY.append(X[1,0])

            var_kal.append(P[0,0])
            last_vx, last_vy = X[2,0], X[3,0]

        print(f"data{count}:", name[index])
        print(f'time is {t}')
        plt.subplot(3,3,draw)
        plt.title(name[index])
        if index == 0: plt.ylabel("path")
        mse.append(round((sum(error_k)/len(error_k)) / (sum(error_o)/len(error_o)), 2))
        
        print(f"mean square error is {mse[-1]}")
        plt.plot(true[:,0],true[:,1], label="true", c='r')
        plt.plot(Z[:,0],Z[:,1],c='orange', label="measure")
        plt.plot(kalmanX, kalmanY, c='g', label="kalman")
        plt.legend()

        plt.subplot(3,3,3+draw)
        if index == 0: plt.ylabel("error")
        plt.plot(range(len(error_obv)), error_obv, c='orange', label="measure")
        plt.plot(range(len(error_kal)), error_kal, c='g', label="kalman")
        plt.legend()

        var_kal = np.array(var_kal) * (1 / max(var_kal))
        
        obv_var = np.array(error_obv).std()**2
        kal_var = np.array(error_kal).std()**2

        plt.subplot(3,3,6+draw)
        if index == 0: plt.ylabel("variance")
        label = f"mean square error={mse[-1]}\n"
        label += f"variance of measure={round(obv_var,2)}\n"
        label += f"variance of kalman={round(kal_var,2)}"

        if draw == 3:
            label += f"    average error = {round(sum(mse)/len(mse),2)}"

        plt.xlabel(label)
        plt.plot(range(len(var_kal)), var_kal, c='g', label="kalman")
        plt.legend()
        print("")
        index += 1

    if save == "true":
        file_name = f'output/data{count}.jpg'
        print(f"Saveing data{count}.jpg...")
        plt.savefig(file_name)
    count += 1
    
if save == "true":
    print("Results have been saved into output/ dir")
    print("")

print("Wait for plot...")
plt.show()


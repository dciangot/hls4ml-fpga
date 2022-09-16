import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import math
import statistics
import matplotlib.pyplot as plt
import scikitplot as skplt


def load_classes(dataset):
    classes = np.load('datasets/'+dataset+'_classes.npy', allow_pickle=True)
    return classes

def read_bm_output(dataset):
    res = []
    with open('outputs/'+dataset+'/bm_output.txt') as f:
        for line in f.readlines():
            splitted = line.replace("[","").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            res.append([float(splitted[0]), float(splitted[1])])
            
    return np.array(res)

def read_software_output(dataset):
    y_keras = np.load("datasets/"+dataset+'_y_keras.npy')
    return y_keras
    
def read_hls4ml_output(dataset):
    res = []
    with open('outputs/'+dataset+'/hls4ml_output.txt') as f:
        for line in f.readlines():
            splitted = line.replace("[","").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            res.append([float(splitted[0]), float(splitted[1])])
            
    return np.array(res)

def load_real_label(dataset):
    y_test = np.load("datasets/"+dataset+'_y_test.npy')
    return y_test

dataset_name = "banknote-authentication"

hls4ml_output = read_hls4ml_output(dataset_name)
bm_output = read_bm_output(dataset_name)

keras_output = read_software_output(dataset_name)[:64]
y_test = load_real_label(dataset_name)[:64]

global_pct_errors_bm_o0 = []
global_pct_errors_hls4ml_o0 = []

global_pct_errors_bm_o1 = []
global_pct_errors_hls4ml_o1 = []

for i in range(0, len(keras_output)):
    #if np.argmax(keras_output[i]) == 1:
    print("software  \t\t\toutput -> ", keras_output[i],     "    classification: ", np.argmax(keras_output[i]))
    print("bondmachine  \t\t\toutput -> ", bm_output[i],        "    classification: ", np.argmax(bm_output[i]))
    print("hls4ml  \t\t\toutput -> ", hls4ml_output[i],    "    classification: ", np.argmax(hls4ml_output[i]))

    p0_keras_output = keras_output[i][0]
    p1_keras_output = keras_output[i][1]
    
    p0_bm_output = bm_output[i][0]
    p1_bm_output = bm_output[i][1]
    
    p0_hls4ml_output = hls4ml_output[i][0]
    p1_hls4ml_output = hls4ml_output[i][1]
    
    pct_error_bm_o0 = (abs(p0_bm_output - p0_keras_output)/p0_keras_output) * 100
    pct_error_bm_o1 = (abs(p1_bm_output - p1_keras_output)/p1_keras_output) * 100
    
    pct_error_hls4ml_o0 = (abs(p0_hls4ml_output - p0_keras_output)/p0_keras_output) * 100
    pct_error_hls4ml_o1 = (abs(p1_hls4ml_output - p1_keras_output)/p1_keras_output) * 100
    
    global_pct_errors_bm_o0.append((pct_error_bm_o0))
    global_pct_errors_hls4ml_o0.append((pct_error_hls4ml_o0))
    
    global_pct_errors_bm_o1.append((pct_error_bm_o1))
    global_pct_errors_hls4ml_o1.append((pct_error_hls4ml_o1))

    print("\n")

print(" Average percentage error classification output 0 for bondmachine ", statistics.mean(global_pct_errors_bm_o0))
print(" Average percentage error classification output 0 for hls4ml      ", statistics.mean(global_pct_errors_hls4ml_o0))
print(" Average percentage error classification output 1 for bondmachine ", statistics.mean(global_pct_errors_bm_o1))
print(" Average percentage error classification output 1 for hls4ml      ", statistics.mean(global_pct_errors_hls4ml_o1))

print("\n")
print("Accuracy keras model (software)     on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(keras_output, axis=1))))
print("Accuracy bondmachine ml on ebaz4205 on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(bm_output, axis=1))))
print("Accuracy hls4ml         on ebaz4205 on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(hls4ml_output, axis=1))))

classes = load_classes(dataset_name)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(np.argmax(y_test, axis=1), np.argmax(hls4ml_output, axis=1))
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
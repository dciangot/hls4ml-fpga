from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
import numpy as np
import os
import tensorflow as tf
import ssl
import sys
import argparse
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import accuracy_score
import hls4ml
import matplotlib.pyplot as plt

class bcolors:
    WHITE = '\033[97m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trainer:

    def __init__(self, fpga_part_number) -> None:
        
        self.fpga_part_number = fpga_part_number
        self.available_datasets = ["mnist_784", "CIFAR_10", "banknote-authentication", "Fashion-MNIST"]
        self.dataset = None
        self.vivado_path = '/tools/Xilinx/Vivado/2019.2/bin:'
        self.seed = 0
        self.le = LabelEncoder()
        self.X_train_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_test = None
        self.model = None
        self.hls_model = None
        self.use_part = True
    
    def initialize(self) -> None:
        
        ssl._create_default_https_context = ssl._create_unverified_context
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ['PATH'] = self.vivado_path + os.environ['PATH']

    def setup_data(self, dataset) -> None:

        if dataset not in self.available_datasets:
            raise Exception("Dataset not available. Availables are: ", self.available_datasets)
        self.dataset = dataset
        file_exists = os.path.exists("datasets/"+dataset+'_X_train_val.npy')
        user_reply = ""
        if file_exists:
            question = bcolors.WARNING + " # QUESTION: Dataset already exists, force re-download? (y/n) "+bcolors.WHITE
            user_reply = input(question)
        else:
            user_reply = "y"
        
        if user_reply == "y":
            data = fetch_openml(dataset)
            x_data, y_data = data['data'], data['target']
            
            x_data.to_csv("datasets/"+dataset+'_raw_x_data.csv', index=False)
            y_data.to_csv("datasets/"+dataset+'_raw_y_data.csv', index=False)

            y = self.le.fit_transform(y_data)
            unique = np.unique(y)
            y = to_categorical(y, len(unique))

            self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(x_data, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            self.X_train_val = scaler.fit_transform(self.X_train_val)
            self.X_test = scaler.transform(self.X_test)
            self.classes = self.le.classes_

            np.save("datasets/"+dataset+'_X_train_val.npy', self.X_train_val)
            np.save("datasets/"+dataset+'_X_test.npy', self.X_test)
            np.save("datasets/"+dataset+'_y_train_val.npy', self.y_train_val)
            np.save("datasets/"+dataset+'_y_test.npy', self.y_test)
            np.save("datasets/"+dataset+'_classes.npy', self.le.classes_)

        else:
            print(bcolors.OKGREEN + " # INFO: Loading dataset", self.dataset+bcolors.WHITE)
            self.X_train_val = np.load("datasets/"+dataset+'_X_train_val.npy')
            self.X_test = np.load("datasets/"+dataset+'_X_test.npy')
            self.y_train_val = np.load("datasets/"+dataset+'_y_train_val.npy')
            self.y_test = np.load("datasets/"+dataset+'_y_test.npy')
            self.classes = np.load("datasets/"+dataset+'_classes.npy', allow_pickle=True)
            

    def exec_train(self):

        file_exists = os.path.exists('models/'+self.dataset+'_KERAS_model.h5')
        user_reply = ""
        if file_exists:
            question = bcolors.WARNING + " # QUESTION: A trained model already exists, do you want to train it again? (y/n) " + bcolors.WHITE
            user_reply = input(question)
        else:
            user_reply = "y"

        if user_reply != "y":
            self.model = load_model('models/'+self.dataset+'_KERAS_model.h5')
            return
        
        unique = np.unique(self.y_train_val)
        classes_len = len(self.classes)

        print(bcolors.OKGREEN + " # Input shape is: "+str(self.X_train_val.shape)+bcolors.WHITE)
        self.model = Sequential()
        self.model.add(Dense(2, input_shape=(self.X_train_val.shape[1],)))
        self.model.add(Activation(activation='relu', name='relu1'))
        self.model.add(Dense(classes_len, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        self.model.add(Activation(activation='softmax', name='softmax'))
        adam = Adam(lr=0.0001)
        self.model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        
        print(bcolors.OKGREEN + " # INFO: Start model training ... "+bcolors.WHITE)
        self.model.fit(self.X_train_val, self.y_train_val, batch_size=int(self.X_train_val.shape[1]/80), epochs=10, validation_split=0.25, shuffle=True)
        
        print(bcolors.OKGREEN + " # INFO: Training finished, saved model path: "+'models/'+self.dataset+'_KERAS_model.h5'+bcolors.WHITE)
        save_model(self.model, 'models/'+self.dataset+'_KERAS_model.h5')

    def exec_test(self):
        print(self.model.summary())
        y_keras = self.model.predict(self.X_test)
        accuracy = format(accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_keras, axis=1)))
        print(bcolors.OKGREEN + " # INFO: Accuracy is "+accuracy+bcolors.WHITE)

    def build_model_fpga(self):
        config = hls4ml.utils.config_from_keras_model(self.model, granularity='Model')
        """ config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<32,8>'
        config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<32,4>'
        for layer in ['fc1', 'relu1', 'output']:
            config['LayerName'][layer]['ReuseFactor'] = 64 """
        
        print(bcolors.OKGREEN + " # INFO: Is using part number "+str(self.use_part)+bcolors.WHITE)
        print(bcolors.OKGREEN + " # INFO: fpga part number     "+self.fpga_part_number+bcolors.WHITE)

        if self.use_part == True:
            self.hls_model = hls4ml.converters.convert_from_keras_model(
                                                        self.model,
                                                        backend='VivadoAccelerator',
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='models_fpga/'+self.dataset+'_hls4ml_prj',
                                                        part=self.fpga_part_number)
        else:
            self.hls_model = hls4ml.converters.convert_from_keras_model(
                                                        self.model,
                                                        backend='VivadoAccelerator',
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='models_fpga/'+self.dataset+'_hls4ml_prj',
                                                        board=self.fpga_part_number)

        self.hls_model.compile()
        supported_boards = hls4ml.templates.get_supported_boards_dict().keys()
        print(bcolors.OKGREEN + " # Supported boards: "+str(supported_boards)+bcolors.WHITE)
        hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config()
        self.hls_model.build(csim=False, synth=True, export=True)
        #hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(self.hls_model)

parser = argparse.ArgumentParser(description="Arguments for training nn", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", help="dataset name")
parser.add_argument("-b", "--fpga_board_number", help="fpga board number")
parser.add_argument("-f", "--fpga_part_number", help="fpga part number")
args = vars(parser.parse_args())
dataset_name = args["dataset"]
fpga_part_number = args["fpga_part_number"]
fpga_board_number = args["fpga_board_number"]

if dataset_name == None or len(dataset_name.replace(" ", "")) == 0:
    print(" # ERROR: No dataset name has been specified. ")
    sys.exit(1)

use_part = True
if fpga_part_number == None and fpga_board_number == None:
    print(bcolors.OKGREEN+" # INFO: FPGA part number not specified, using default xc7z010clg400-1"+bcolors.WHITE)
    fpga_part_number = "xc7z010clg400-1"
elif fpga_board_number != None:
    fpga_part_number = fpga_board_number
    use_part = False

t = Trainer(fpga_part_number)
t.use_part = use_part
t.initialize()
try:
    t.setup_data(dataset_name)
except Exception as e:
    print(" # An error occurred during setup data:", e)
    sys.exit(1)

t.exec_train()
t.exec_test()
t.build_model_fpga()
sys.exit()

'''
Each image of the MNIST dataset is encoded in a 784 dimensional vector, 
representing a 28 x 28 pixel image. Each pixel has a value between 0 and 255, 
corresponding to the grey-value of a pixel
'''
 
datasets = ["mnist_784", "CIFAR_10"]
 
download_data = False
 
ssl._create_default_https_context = ssl._create_unverified_context
 
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
 
if download_data:
    data = fetch_openml('CIFAR_10')
    X, y = data['data'], data['target']
 
    print(data['feature_names'])
    print(X.shape, y.shape)
    print(X[:5])
    print(y[:5])
 
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 10)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y[:5])
 
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)
 
    np.save('X_train_val.npy', X_train_val)
    np.save('X_test.npy', X_test)
    np.save('y_train_val.npy', y_train_val)
    np.save('y_test.npy', y_test)
    np.save('classes.npy', le.classes_)
 
X_train_val = np.load("X_train_val.npy")
X_test = np.load("X_test.npy")
y_train_val = np.load("y_train_val.npy")
y_test = np.load("y_test.npy")
classes = np.load("classes.npy",allow_pickle=True)
 
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
 
model = Sequential()
# 3072 is computed for cifar_10 by 32x32 (width and height of images) multily by 3 which are the color channels 
model.add(Dense(64, input_shape=(3072,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu1'))
model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu2'))
model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu3'))
model.add(Dense(10, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='softmax', name='softmax'))
 
train = True
if train:
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=15, validation_split=0.25, shuffle=True)
else:
    from tensorflow.keras.models import load_model
    model = load_model('model_1/KERAS_check_best_model.h5')
 
import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
 
y_keras = model.predict(X_test)
print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
plt.figure(figsize=(9,9))
_ = plotting.makeRoc(y_test, y_keras, classes)
plt.show()
 
# xc7z010clg400-1
 
import hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='model_1/hls4ml_prj',
                                                       part='xc7z010clg400-1')
 
hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
hls_model.compile()
X_test = np.ascontiguousarray(X_test)
y_hls = hls_model.predict(X_test)
 
print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
 
fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.makeRoc(y_test, y_keras, classes)
plt.gca().set_prop_cycle(None) # reset the colors
_ = plotting.makeRoc(y_test, y_hls, classes, linestyle='--')
 
from matplotlib.lines import Line2D
lines = [Line2D([0], [0], ls='-'),
         Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend
leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
            loc='lower right', frameon=False)
ax.add_artist(leg)
 
hls_model.build(csim=False)
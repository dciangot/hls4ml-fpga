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
        self.available_datasets = [
            "mnist_784", 
            "CIFAR_10", 
            "banknote-authentication",
            "Fashion-MNIST",
            "hls4ml_lhc_jets_hlf",
            "shuttle-landing-control",
            "climate-model-simulation-crashes"]
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
        self.classes_len = 0
    
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
            

    def build_model(self, m_type):
        if m_type == "MLP":
            self.model = Sequential()
            self.model.add(Dense(1, input_shape=(self.X_train_val.shape[1],), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
            self.model.add(Activation(activation='relu', name='relu1'))
            self.model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
            self.model.add(Activation(activation='relu', name='relu2'))
            # self.model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
            # self.model.add(Activation(activation='relu', name='relu3'))
            self.model.add(Dense(self.classes_len, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
            self.model.add(Activation(activation='softmax', name='softmax'))

            adam = Adam(lr=0.0001)
            self.model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        
        elif m_type == "CNN":
            
            from tensorflow.keras import layers
            
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dense(self.classes_len))
            
            adam = Adam(lr=0.0001)
            self.model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

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
        self.classes_len = len(self.classes)

        print(bcolors.OKGREEN + " # Input shape is: "+str(self.X_train_val.shape)+bcolors.WHITE)

        self.build_model("CNN")
        
        print(bcolors.OKGREEN + " # INFO: Start model training ... "+bcolors.WHITE)
        self.model.fit(self.X_train_val, self.y_train_val, batch_size=int(self.X_train_val.shape[1]*10), epochs=10, validation_split=0.25, shuffle=True)
        
        print(bcolors.OKGREEN + " # INFO: Training finished, saved model path: "+'models/'+self.dataset+'_KERAS_model.h5'+bcolors.WHITE)
        save_model(self.model, 'models/'+self.dataset+'_KERAS_model.h5')

    def exec_test(self):
        print(self.model.summary())
        input()
        y_keras = self.model.predict(self.X_test)
        accuracy = format(accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_keras, axis=1)))
        print(bcolors.OKGREEN + " # INFO: Accuracy is "+accuracy+bcolors.WHITE)

    def build_model_fpga(self):
        config = hls4ml.utils.config_from_keras_model(self.model, granularity='name')
        """ config['LayerName']['fc1']['exp_table_t'] = 'ap_fixed<8,6>'
        config['LayerName']['fc1']['inv_table_t'] = 'ap_fixed<8,6>'
        config['LayerName']['fc2']['exp_table_t'] = 'ap_fixed<8,6>'
        config['LayerName']['fc2']['inv_table_t'] = 'ap_fixed<8,6>'
        config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<8,6>'
        config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<8,6>' """
        """ for layer in ['relu1', 'softmax']:
            config['LayerName'][layer]['ReuseFactor'] = 784 """
        
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
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.regularizers import l1
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
import sys
import argparse
from sklearn.metrics import accuracy_score
import hls4ml
import matplotlib.pyplot as plt
import json
from os.path import exists

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

    def __init__(self, fpga_part_number, nn_model_type):
        
        self.nn_model_type = nn_model_type
        self.fpga_part_number = fpga_part_number
        self.available_datasets = [
            "mnist_784", 
            "CIFAR_10", 
            "banknote-authentication",
            "Fashion-MNIST",
            "hls4ml_lhc_jets_hlf",
            "shuttle-landing-control",
            "climate-model-simulation-crashes",
            "monks-problems-2",
            "diabetes",
            "pc4",
            "madelon",
            "scene", 
            "PhishingWebsites",
            "optdigits",
            "higgs",
            "dna"]
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
        self.train_size = 0
        self.network_spec = None

    def initialize(self) -> None:
        
        ssl._create_default_https_context = ssl._create_unverified_context
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
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
            

    def parse_network_specifics(self):
        
        try:
            f = open('specifics.json')
            self.network_spec = json.load(f)
        except:
            return

    def build_model(self):
        if self.nn_model_type == "MLP":

            self.model = Sequential()

            self.parse_network_specifics()

            if self.network_spec == None:
                self.model.add(Dense(4, input_shape=(self.X_train_val.shape[1],), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                self.model.add(Activation(activation='relu'))
                self.model.add(Dense(8, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                self.model.add(Activation(activation='relu'))
                opt = Adam(lr=0.0001)
            else:
                arch = self.network_spec["network"]["arch"]
                for i in range(0, len(arch)):
                    layer_name = self.network_spec["network"]["arch"][i]["layer_name"]
                    activation_function = self.network_spec["network"]["arch"][i]["activation_function"]
                    neurons = self.network_spec["network"]["arch"][i]["neurons"]
                    if i == 0:
                        self.model.add(Dense(neurons, input_shape=(self.X_train_val.shape[1],), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                    else:
                        self.model.add(Dense(neurons, name=layer_name, kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                    self.model.add(Activation(activation=activation_function))

                if  self.network_spec["network"]["training"]["optimizer"] == "Adam":
                    opt = Adam(lr=0.0001)
                elif self.network_spec["network"]["training"]["optimizer"] == "Adagrad":
                    opt = Adagrad(lr=0.0001)
                else:
                    opt = Adam(lr=0.0001)
                # handle more opt
                
            self.model.add(Dense(self.classes_len, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
            self.model.add(Activation(activation='softmax'))

            self.model.compile(optimizer=opt, loss=['categorical_crossentropy'], metrics=['accuracy'])

        # WIP: to handle more model type

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
        self.train_size = self.X_train_val.shape[0]

        self.build_model()

        checkpoint_path = 'models/'+self.dataset+'/training/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

        if self.network_spec == None:
            print(bcolors.OKGREEN + " # INFO: Start model training with networks specs ... "+bcolors.WHITE)
            self.model.fit(
                self.X_train_val, 
                self.y_train_val, 
                batch_size=int(self.X_train_val.shape[1]*10), 
                epochs=20, 
                validation_split=0.25, 
                shuffle=True,
                callbacks=[cp_callback])
        else:
            print(bcolors.OKGREEN + " # INFO: Start model training with networks specs ... "+bcolors.WHITE)
            batch_size = int(self.X_train_val.shape[1]*10) if self.network_spec["network"]["training"]["batch_size"] == "default" else int(self.network_spec["network"]["training"]["batch_size"])
            epochs = int(self.network_spec["network"]["training"]["epochs"])
            validation_split = 0.25 if self.network_spec["network"]["training"]["validation_split"] == "default" else float(self.network_spec["network"]["training"]["validation_split"])
            shuffle = True if self.network_spec["network"]["training"]["shuffle"] == "true" else False

            self.model.fit(
                self.X_train_val, 
                self.y_train_val, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_split=validation_split, 
                shuffle=shuffle,
                callbacks=[cp_callback])
        
        tf.keras.utils.plot_model(
            self.model,
            to_file='models/'+self.dataset+'/model.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96,
            layer_range=None,
            show_layer_activations=True
        )
        print(bcolors.OKGREEN + " # input name", self.model.input.op.name+bcolors.WHITE)
        print(bcolors.OKGREEN + " # output name", self.model.output.op.name+bcolors.WHITE)
        print(bcolors.OKGREEN + " # INFO: Training finished, saved model path: "+'models/'+self.dataset+'_KERAS_model.h5'+bcolors.WHITE)
        self.model.save('models/'+self.dataset+'/model.h5')
        meta_graph_def = tf.train.export_meta_graph(filename='models/'+self.dataset+'_KERAS_model.meta')

    def exec_test(self):
        print(self.model.summary())
        input(bcolors.OKGREEN+" # INFO: press enter to continue"+bcolors.WHITE)
        y_keras = self.model.predict(self.X_test)
        accuracy = format(accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_keras, axis=1)))
        print(bcolors.OKGREEN + " # INFO: Accuracy is "+accuracy+bcolors.WHITE)

    def build_model_fpga(self):
        config = hls4ml.utils.config_from_keras_model(self.model, granularity='name')
        
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
parser.add_argument("-m", "--nn_model_type", help="neural network architecture")
args = vars(parser.parse_args())
dataset_name = args["dataset"]
fpga_part_number = args["fpga_part_number"]
fpga_board_number = args["fpga_board_number"]
nn_model_type = args["nn_model_type"]

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

if nn_model_type == None:
    nn_model_type = "MLP"

t = Trainer(fpga_part_number, nn_model_type)
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
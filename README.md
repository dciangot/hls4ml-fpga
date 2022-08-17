
<div align="center"> 
    <h2 align="center">Machine Learning with HLS4ML&PYNQ on EBAZ4205 FPGA!</h2>
    <img src="/images/0_ebazlogo.png" alt="EBAZ4205" style="display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;"/> 
</div>

##### This is a step by step tutorial on how to deploy a simple MLP neural network on EBAZ4205 fpga.
The following libraries and tools have been used in order to achieve the goal:

- [hls4ml][1]
- [tensorflow][2]
- [pynq][3]
- [vivado][4]
- [conda][5]

This tutorial will be divided in two parts: in the first one, you will generate the IP using the hls4ml library which is the core part of the inference phase on the FPGA. The IP will be a part of the overall design which is necessary to generate the bistream that runs on the FPGA. 
At the end of this tutorial you will have all the necessary files to try and test the DNN model on the ebaz with the help of PYNQ that provides the API to access and use the bitstream (second part).
This guide does not focus on evaluating the accuracy performance of the neural network model or the speed of inference. The focus is on how to deploy a neural network on an FPGA that was previously used as control card of Ebit E9 + BTC miner.

[1]: https://github.com/fastmachinelearning/hls4ml
[2]: https://www.tensorflow.org/
[3]: http://www.pynq.io/
[4]: https://www.xilinx.com/products/design-tools/vivado.html
[5]: https://docs.conda.io/en/latest/
[6]: https://www.openml.org/
[7]: https://www.openml.org/search?type=data&status=active&id=42468

##### STEP 0: Check requirements
First of all you need a basic knowledge of using Vivado and programming in Python.
This guide has been developed on **Ubuntu 20.04** and it is necessary to install **Vivado 2019.2** in order to make the **bitstream** that will run on FPGA. 
Furthermore, on the EBAZ4205 is running a custom linux-based image that runs **PYNQ**. This image can be found in this github repo: https://github.com/Stavros/ebaz4205 (The link to directly download it is: https://drive.google.com/file/d/1MnXFLagFiFrE1o9HDPSr34sZ4QsQKgjx/view).
Flash the IMG file on a sd-card and the EBAZ4205 is ready to run the machine learning bitstream. But before is obviously necessary to make the bitstream.

##### STEP 1: Clone this repo and setup environment with conda
Simply run the following commands to setup the environment and install the necessary packages
```
git clone https://github.com/Bianco95/hls4ml-fpga.git
conda create --name hls4ml-env python=3.8.0
conda activate hls4ml-env
pip3 install -r requirements.txt
```

##### STEP 2: Make the ML IP 
Run the following command to build the IP:
```
python3 main.py --dataset hls4ml_lhc_jets_hlf -m MLP
```
As you can see from the command above, it is necessary to specify the dataset you want to train the neural network with. 
Datasets are fetched inside the code thanks to [openML][6].
Here the dataset specified is **hls4ml_lhc_jets_hlf** whose specifications can be found on this [link][7].
You can also specify the type of neural network (at the moment is only supported a simple neural network with a bunch of dense layer).
If the dataset has already been downloaded, you will be asked if you want to download it again. The same is for the neural network model, if it already exists you will be asked if you want to train it again. 
In the near future more commands line options will be supported, for example training epochs, batch size, DNN architecture (i.e. number of layers and nodes, activation functions, ...).
The default model is a MLP sequential model with one hidden layer, this means three layers in total (input layer, hidden dense layer, output layer).
The neural network model architecture is very basic, the goal of this guide is to deploy the ML model on the FPGA and moreover the resources of the EBAZ4205 are very limited. In fact, the FPGA resources in terms of **LUTS**, **BRAM** and **FLIPFLOP** essentially depend on two factors: the complexity of the model and the **number of features** of the dataset. Regarding the latter, a study was carried out on the occupation of FPGA resources using the same neural network as the dataset varies, which has shown that the occupation of FPGA resources grows with respect to the number of features.
Wait for the command to complete. If all went well, under the folder **_models_fpga/hls4ml_lhc_jets_hlf_hls4ml_prj/myproject_prj/solution1/impl_** there will be the newly created ip.

##### STEP 3: Use vivado for final design and bitstream 
Open vivado and create a new project specifying the part number of the ebaz4205 (*xc7z010clg400-1*) as shown in the image below.

<img src="/images/1_setup.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%;"/> 
Then, go to **Tools** -> **Settings** and under **IP** -> **Repository** click the plus button and add the absolute path for **_models_fpga/hls4ml_lhc_jets_hlf_hls4ml_prj/myproject_prj/solution1/impl_**. Vivado will detect the IP (if there is one) as show in the imabe below:

<img src="/images/2_addip.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%;"/> 
In this github repo, under the *resources* directory there is a block design file fully functional and ready to use, just import it and that's it.
Go to **Ip Integrator** -> **Import block design** and select the block design in the resources directory.
The block design should look like the following:

<img src="/images/6_bd.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 65%;"/>

As you can see, the overall design is already complete and ready to be used. All components are configured and suitable for generating a bitstream that works for ebaz4205, but you can also build your own design for another ZYNQ-FPGA starting from mine.

Import the **constraint** file for the ebaz4205 as show in the image below:

<img src="/images/3_addconst.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%;"/> 
Then, right click on **design_1_i** and select **Create HDL Wrapper**. 

<img src="/images/4_hdlwrapper.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%;"/> 
Finally, generate the bitstream using the button show in the figure below:

<img src="/images/5_bitstream.png" alt="EBAZ4205" style="display: block;
    margin-left: auto;
    margin-right: auto;
    width: 60%;"/> 
This will run the synthesis, implementation and bitstream code generation.

##### STEP 4: Copy the generated files on EBAZ4205
After the bitstream generation, a file with the *.bit* extension will be generated under *project-folder-name/project-folder-name.runs/impl_1/* (in my case the file's name is *design_1.bit*).
Copy this file inside the jupyter notebook directory on the FPGA (ebaz must be connected to the network or at least reachable in your local network).
```
scp design_1_wrapper.bit xilinx@10.2.1.103:/home/xilinx/jupyter_notebooks/nn_mlp.bit
```
Moreover, you have to copy the hardware file (*.hwh*) inside the same directory and it must have the same name as the *.bit* file. This file is located under *project-folder-name/project-folder-name.srcs/sources_1/bd/design_1/hw_handoff/*
```
scp design_1.hwh xilinx@10.2.1.103:/home/xilinx/jupyter_notebooks/nn_mlp.hwh
```
Finally, copy the test datasets under the *datasets* folder in this repo and the *axi_stream_driver.py*
```
scp datasets/hls4ml_lhc_jets_hlf_y_testt.npy xilinx@10.2.1.103:/home/xilinx/jupyter_notebooks/
scp datasets/hls4ml_lhc_jets_hlf_X_test.npy xilinx@10.2.1.103:/home/xilinx/jupyter_notebooks/
scp models_fpga/hls4ml_lhc_jets_hlf_hls4ml_prj/axi_stream_driver.py xilinx@10.2.1.103:/home/xilinx/jupyter_notebooks/
```
For the second part, check the notebook called **1_NN.ipynb** under the *notebooks* directory.





# neural-planes

## Why it exists?
This programm was my experiment in Deep Learning technology. 

My task was to solve the detection an classification problem at the same time on aerial objects:

1. Birds (2 types)
2. Army aeroplanes (3 types)
3. Civil aeroplanes (3 types)

Each of these objects was represented on the image with different backgrounds, wich representing 
different weather condidtions.

For solving this problem of was using modified and deeply simplified version of [YOLO](https://arxiv.org/abs/1506.02640) architecture.

## Some results

Civil plane class with probability 0.4:

![alt text](https://github.com/dkaravaev/neural-planes/blob/master/results/civil_pr40.png)

Fighter plane class with probability 0.47:

![alt text](https://github.com/dkaravaev/neural-planes/blob/master/results/fighter_pr47.png)

Fighter plane class with probability 0.58:

![alt text](https://github.com/dkaravaev/neural-planes/blob/master/results/fighter_pr58.png)

## Architecture

There are 3 independent modules of programm. The settings of their work are written in unified config.json.

1. Gendata:
This module is for generating data for training the neural net. The output data is represented as images and XML annotations.
It's written in С++, because the programm is generating data from 3D models, and the simpliest way of 
manipulating with them on Linux was OpenSceneGraph. The programm runs in CLI mode with config filename as command line argument.
2. Archdata:
This module is for archiving data and it's written in Python. This module is translates the images and annotations to HDF5 archives
with Numpy arrays in the form of dirrect input-ouput of neural net.
3. Network:
The last but not least. This is the main module. It's written in Python with stack Keras/Theano/cuDNN/CUDA. It just simply has object Network,
which you can train or use it for predictions.

## Requirements
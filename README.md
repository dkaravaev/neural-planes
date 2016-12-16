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

All settings of programm are written in config.json. 

1. Gendata
2. Archdata
3. Network
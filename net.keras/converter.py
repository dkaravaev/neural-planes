import os
import numpy as np

from math import pow

from PIL import Image
from PIL import ImageDraw


class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes,1))


def convert_yolo_detections(predictions,classes=20,num=2,side=7,w=1,h=1,threshold=0.2,only_objectness=0):
    boxes = []
    probs = np.zeros((side*side*num,classes))
    for i in range(side*side):
        row = i / side
        col = i % side
        for n in range(num):
            index = i*num+n
            p_index = side*side*classes+i*num+n
            scale = predictions[p_index]
            box_index = side*side*(classes+num) + (i*num+n)*4

            new_box = box(classes)
            new_box.x = (predictions[box_index + 0] + col) / side * w
            new_box.y = (predictions[box_index + 1] + row) / side * h
            new_box.h = pow(predictions[box_index + 2], 2) * w
            new_box.w = pow(predictions[box_index + 3], 2) * h

            for j in range(classes):
                class_index = i*classes
                prob = scale*predictions[class_index+j]
                if(prob > threshold):
                    new_box.probs[j] = prob
                else:
                    new_box.probs[j] = 0
            if(only_objectness):
                new_box.probs[0] = scale

            boxes.append(new_box)
    return boxes


def prob_compare(boxa,boxb):
    if(boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]):
        return 1
    elif(boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]):
        return 0
    else:
        return -1


def do_nms_sort(boxes,total,classes=20,thresh=0.5):
    sorted_boxes = []
    for k in range(classes):
        for box in boxes:
            box.class_num = k
        sorted_boxes = boxes.sort(cmp=prob_compare)
        for i in range(total):
            if(sorted_boxes[i].probs[k] == 0):
                continue
            boxa = sorted_boxes[i]
            for j in range(i+1,total):
                boxb = sorted_boxes[j]
                if(boxb.probs[k] != 0 and box_iou(boxa,boxb) > thresh):
                    boxb.probs[k] = 0
                    sorted_boxes[j] = boxb
    return sorted_boxes


def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

def draw_detections(impath,num,thresh,boxes,classes,labels,save_name):
    """
    Args:
        impath: The image path
        num: total number of bounding boxes
        thresh: boxes prob beyond this thresh will be drawn
        boxes: boxes predicted by the network
        classes: class numbers of the objects
    """
    img = Image.open(impath)
    drawable = ImageDraw.Draw(img)
    ImageSize = img.size
    for i in range(num):
        #for each box, find the class with maximum prob
        max_class = np.argmax(boxes[i].probs)
        prob = boxes[i].probs[max_class]
        if(prob > thresh):
            b = boxes[i]

            temp = b.w
            b.w = b.h
            b.h = temp

            left  = (b.x-b.w/2.)*ImageSize[0];
            right = (b.x+b.w/2.)*ImageSize[0];
            top   = (b.y-b.h/2.)*ImageSize[1];
            bot   = (b.y+b.h/2.)*ImageSize[1];

            if(left < 0): left = 0;
            if(right > ImageSize[0]-1): right = ImageSize[0]-1;
            if(top < 0): top = 0;
            if(bot > ImageSize[1]-1): bot = ImageSize[1]-1;

            print("The four cords are: ",left,right,top,bot)
            drawable.rectangle([left,top,right,bot],outline="red")
            img.save(os.path.join(os.getcwd(),'results',save_name))
            print(labels[max_class],": ",boxes[i].probs[max_class])

# draw_detections(os.path.join(imagePath, image_name),98,0.2, boxes, 20,labels, image_name)

import theano
import numpy

from theano import tensor as T


def overlap(x1,w1,x2,w2,side=7):
    """
    Args:
      x1: x for the first box
      w2: width for the first box
      x2 : x for the second box
      w2: width for the second box
    """
    #Find out how much percent the center is over the whole width/height
    x1 = x1 / side
    x2 = x2 / side
    l1 = x1 - w1*w1/2
    l2 = x2 - w2*w2/2
    left = T.switch(T.lt(l1,l2),l2,l1)
    r1 = x1 + w1*w1/2
    r2 = x2 + w2*w2/2
    right = T.switch(T.lt(r1,r2),r1,r2)
    return right - left


def box_intersection(a,b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the intersection area of each two boxes
    """
    w = overlap(a[:,:,0],a[:,:,3],b[:,:,0],b[:,:,3])
    h = overlap(a[:,:,1],a[:,:,2],b[:,:,1],b[:,:,2])
    w = T.switch(T.lt(w,0),0,w)
    h = T.switch(T.lt(h,0),0,h)
    area = w*h
    return area


def box_union(a,b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the union area of each two boxes
    """
    i = box_intersection(a,b)
    area_a = a[:,:,2]*a[:,:,3]
    area_b = b[:,:,2]*b[:,:,3]
    u = area_a*area_a + area_b*area_b - i
    return u


def box_iou(a,b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the intersection over union of each two boxes
    Notes: boxes with 0 union should has 0 iou, don't know why orignal yolo ignores this situation
    """
    #the net and groud truth are all the square root of height and width
    u = box_union(a,b)
    i = box_intersection(a,b)
    iou = T.switch(T.eq(u,0),0,i/u)
    return iou


def box_mse(a,b):
    mse = T.sum(T.square(a-b),axis=2)
    return mse

"""
YOLO Input: Image in RGB
YOLO Output:
    SIDE x SIDE x CLASSES:
        P_{ijk}
        - Probability of ij-cell has object with k-class
    SIDE x SIDE x B:
        scale_{ij0} ... scale_{ijB}
        - Class scales for each bounding box in ij-cell
    SIDE x SIDE x B x 4:
        (x_{ij0}, y_{ij0}, sqrt(h_{ij0}), sqrt(w_{ij0})) ... (x_{ijB}, y_{ijB}, sqrt(h_{ijB}), sqrt(w_{ijB}))
        - Bounding box definition in each ij-cell
"""
SIDE = 7
B = 2
C = 20


def custom_loss_2(y_true, y_pred):
    """
    Args:
      y_true: ground truth tensor
      y_pred: tensor predicted by the network
    Returns:
      loss: the summed mse loss, details of this loss function can be found in yolo paper
    """
    # the ground truth x is the offset within a cell, but we need actual ground truth x to calculate the overlap
    # MAGIC NUMBERS!
    offset = []
    for i in range(SIDE):
        for j in range(SIDE):
            offset.append(j)
            offset.append(i)
            offset.extend([0]*3)
            offset.append(j)
            offset.append(i)
            offset.extend([0]*23)
    y_pred_offset = y_pred + numpy.asarray(offset)
    y_true_offset = y_true + numpy.asarray(offset)

    loss = 0.0
    y_pred = y_pred.reshape((y_pred.shape[0], SIDE * SIDE, B * 5 + C))
    y_true = y_true.reshape((y_true.shape[0], SIDE * SIDE, B * 5 + C))
    y_pred_offset = y_pred_offset.reshape((y_pred_offset.shape[0], SIDE * SIDE, B * 5 + C))
    y_true_offset = y_true_offset.reshape((y_true_offset.shape[0], SIDE * SIDE, B * 5 + C))

    # Get two bounding boxes
    a_offset = y_pred_offset[:, :, 0:4]
    b_offset = y_pred_offset[:, :, 5:9]
    gt_offset = y_true_offset[:, :, 0:4]

    a = y_pred[:, :, 0:4]
    b = y_pred[:, :, 5:9]
    gt = y_true[:, :, 0:4]

    # iou between box a and gt
    iou_a_gt = box_iou(a_offset, gt_offset)
    # don't want iou has influence on x,y,h,w, x,y,h,w are only infected by gt value
    iou_a_gt = theano.gradient.disconnected_grad(iou_a_gt)

    # iou between box b and gt
    iou_b_gt = box_iou(b_offset, gt_offset)
    iou_b_gt = theano.gradient.disconnected_grad(iou_b_gt)

    # mse between box a and gt
    mse_a_gt = box_mse(a_offset, gt_offset)

    # mse between box b and gt
    mse_b_gt = box_mse(b_offset, gt_offset)

    # mask is either 0 or 1, 1 indicates box b has a higher iou with gt than box a
    mask = T.switch(T.lt(iou_a_gt, iou_b_gt), 1, 0)

    # if two boxes both have 0 iou with ground truth, we blame the one with higher mse with gt
    # It feels like hell to code like this,f**k!
    mask_iou_zero = T.switch(T.and_(T.le(iou_a_gt, 0), T.le(iou_b_gt, 0)), 1, 0)
    mask_mse = T.switch(T.lt(mse_a_gt, mse_b_gt), 1, 0)
    mask_change = mask_iou_zero * mask_mse
    mask = mask + mask_change
    mask = theano.gradient.disconnected_grad(mask)

    # loss between box a and gt
    # 5 for \lambda_{obj}
    loss_a_gt = T.sum(T.square(a - gt), axis=2) * 5

    # loss between box b and gt
    loss_b_gt = T.sum(T.square(b - gt), axis=2) * 5

    # use mask to add the loss from the box with higher iou with gt
    loss += y_true[:, :, 4] * (1 - mask) * loss_a_gt
    loss += y_true[:, :, 4] * mask * loss_b_gt

    # confident loss between a and gt
    closs_a_gt = T.square(iou_a_gt * y_true[:, :, 4] - y_pred[:, :, 4])
    # confident loss between b and gt
    closs_b_gt = T.square(iou_b_gt * y_true[:, :, 4] - y_pred[:, :, 9])

    loss += closs_a_gt * (1-mask) * y_true[:, :, 4]
    loss += closs_b_gt * mask * y_true[:, :, 4]

    # if the cell has no obj, confidence loss should be halved
    # 0.5 = \lambda_{noobj}
    loss += closs_a_gt * (1-y_true[:, :, 4]) * 0.5
    loss += closs_b_gt * (1-y_true[:, :, 4]) * 0.5

    # add loss for the conditioned classification error
    # Probs for classes
    loss += T.sum(T.square(y_pred[:, :, 10:30] - y_true[:, :, 10:30]), axis=2) * y_true[:, :, 4]

    # sum for each cell
    loss = T.sum(loss)

    # mean for each image
    # loss = T.mean(loss)

    return loss


def convert_yolo_detections(predictions,classes=20,num=2,square=True,side=7,w=1,h=1,threshold=0.2,only_objectness=0):
    boxes = []
    probs = numpy.zeros((side*side*num,classes))
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
import theano
import numpy

from yolo_loss import geometry
from theano import tensor as T

SIDE = 7
B = 2
CLASSES = 3
LAMBDA_OBJ = 5
LAMBDA_NOOBJ = .5

"""
INFO:
    YOLO Input: Image in RGB
    YOLO Output:
        SIDE x SIDE x CLASSES:
            P_{ijk}(Class_{i}|Object)
            - Probability of ij-cell has object with k-class
        SIDE x SIDE x B:
            P_{ij0}(Object) ... P_{ijB}(Object)
            - Probability of ij has some Object.
        SIDE x SIDE x B x 4:
            (x_{ij0}, y_{ij0}, sqrt(h_{ij0}), sqrt(w_{ij0})) ... (x_{ijB}, y_{ijB}, sqrt(h_{ijB}), sqrt(w_{ijB}))
            - Bounding box definition in each ij-cell
    NOTE: Confidence
"""
"""
def function(y_true, y_pred):
    y1 = y_pred
    y2 = y_true
    loss = 0.0

    scale_vector = []
    scale_vector.extend([2]*4)
    scale_vector.extend([1]*20)
    scale_vector = numpy.reshape(numpy.asarray(scale_vector),(1,len(scale_vector)))

    for i in range(49):
        y1_piece = y1[:,i*25:i*25+24]
        y2_piece = y2[:,i*25:i*25+24]

        y1_piece = y1_piece * scale_vector
        y2_piece = y2_piece * scale_vector

        loss_piece = T.sum(T.square(y1_piece - y2_piece),axis=1)
        loss = loss + loss_piece * y2[:,i*25+24]
        loss = loss + T.square(y2[:,i*25+24] - y1[:,i*25+24])

    #loss = T.sum(loss)
    loss = T.sum(loss)
    return loss

def function1(y_true, y_pred):
    offset = numpy.zeros(49 * (3 + 5 * 2))
    for i in range(SIDE * SIDE):
        row = i / SIDE
        col = i % SIDE

        box_index = SIDE * SIDE * (CLASSES + B) + i * B * 4
        offset[box_index + 0] = col
        offset[box_index + 1] = row

    y_pred_offset = y_pred + offset
    y_true_offset = y_true + offset

    loss = 0.0
    y_pred = y_pred.reshape((y_pred.shape[0], SIDE * SIDE, B * 5 + CLASSES))
    y_true = y_true.reshape((y_true.shape[0], SIDE * SIDE, B * 5 + CLASSES))
    y_pred_offset = y_pred_offset.reshape((y_pred_offset.shape[0], SIDE * SIDE, B * 5 + CLASSES))
    y_true_offset = y_true_offset.reshape((y_true_offset.shape[0], SIDE * SIDE, B * 5 + CLASSES))

    a_offset = y_pred_offset[:, :, 0:4]
    b_offset = y_pred_offset[:, :, 5:9]
    gt_offset = y_true_offset[:, :, 0:4]

    a = y_pred[:, :, 0:4]
    b = y_pred[:, :, 5:9]
    gt = y_true[:, :, 0:4]

    iou_a_gt = geometry.box_iou(a_offset, gt_offset)
    iou_a_gt = theano.gradient.disconnected_grad(iou_a_gt)

    iou_b_gt = geometry.box_iou(b_offset, gt_offset)
    iou_b_gt = theano.gradient.disconnected_grad(iou_b_gt)

    mse_a_gt = geometry.box_mse(a_offset, gt_offset)
    mse_b_gt = geometry.box_mse(b_offset, gt_offset)

    mask = T.switch(T.lt(iou_a_gt, iou_b_gt), 1, 0)
    mask_iou_zero = T.switch(T.and_(T.le(iou_a_gt, 0), T.le(iou_b_gt, 0)), 1, 0)
    mask_mse = T.switch(T.lt(mse_a_gt, mse_b_gt), 1, 0)
    mask_change = mask_iou_zero * mask_mse
    mask = mask + mask_change
    mask = theano.gradient.disconnected_grad(mask)

    loss_a_gt = T.sum(T.square(a - gt), axis=2) * LAMBDA_OBJ
    loss_b_gt = T.sum(T.square(b - gt), axis=2) * LAMBDA_OBJ

    loss += y_true[:, :, 4] * (1 - mask) * loss_a_gt
    loss += y_true[:, :, 4] * mask * loss_b_gt

    closs_a_gt = T.square(iou_a_gt * y_true[:, :, 4] - y_pred[:, :, 4])
    closs_b_gt = T.square(iou_b_gt * y_true[:, :, 4] - y_pred[:, :, 9])

    loss += closs_a_gt * (1 - mask) * y_true[:, :, 4]
    loss += closs_b_gt * mask * y_true[:, :, 4]

    loss += closs_a_gt * (1 - y_true[:, :, 4]) * LAMBDA_NOOBJ
    loss += closs_b_gt * (1 - y_true[:, :, 4]) * LAMBDA_NOOBJ

    loss += T.sum(T.square(y_pred[:, :, B * 5:B * 5 + CLASSES] - y_true[:, :, B * 5:B * 5 + CLASSES]), axis=2) * y_true[:, :, 4]

    print(loss)
    loss = T.sum(loss)
    return loss
"""

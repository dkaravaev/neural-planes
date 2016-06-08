import numpy

from theano import tensor, gradient

side = 7
boxes = 2
classes = ["fighter", "civil-plane", "bird"]


class SingleDetectionLoss:
    PENALTY_OBJ = 4

    @staticmethod
    def function(y_true, y_pred):
        loss = 0.0

        scale_vector = numpy.asarray([SingleDetectionLoss.PENALTY_OBJ] * 4)
        for i in range(49):
            box_true = y_true[:, i * 8:i * 8 + 4]
            box_pred = y_pred[:, i * 8:i * 8 + 4]

            has_obj = y_true[:, i * 8 + 7]
            pred_conf = y_pred[:, i * 8 + 7]

            probs_true = y_true[:, i * 8 + 4:i * 8 + 7]
            probs_pred = y_pred[:, i * 8 + 4:i * 8 + 7]

            box_loss = tensor.sum(tensor.mul(scale_vector, tensor.square(box_true - box_pred)), axis=1)
            prob_loss = tensor.sum(tensor.square(probs_true - probs_pred), axis=1)

            loss += tensor.mul(has_obj, box_loss)
            loss += tensor.mul(has_obj, prob_loss)
            loss += tensor.square(has_obj - pred_conf)

        loss = tensor.sum(loss)
        return loss


class MultiDetectionLoss:
    @staticmethod
    def overlap(x1, w1, x2, w2):
        x1 /= side
        x2 /= side
        l1 = x1 - w1*w1/2
        l2 = x2 - w2*w2/2
        left = tensor.switch(tensor.lt(l1, l2), l2, l1)
        r1 = x1 + w1*w1/2
        r2 = x2 + w2*w2/2
        right = tensor.switch(tensor.lt(r1, r2), r1, r2)
        return right - left

    @staticmethod
    def box_intersection(a, b):
        w = MultiDetectionLoss.overlap(a[:, :, 0], a[:, :, 3], b[:, :, 0], b[:, :, 3])
        h = MultiDetectionLoss.overlap(a[:, :, 1], a[:, :, 2], b[:, :, 1], b[:, :, 2])
        w = tensor.switch(tensor.lt(w, 0), 0, w)
        h = tensor.switch(tensor.lt(h, 0), 0, h)
        area = w * h
        return area

    @staticmethod
    def box_union(a, b):
        i = MultiDetectionLoss.box_intersection(a, b)
        area_a = a[:, :, 2] * a[:, :, 3]
        area_b = b[:, :, 2] * b[:, :, 3]
        u = area_a*area_a + area_b*area_b - i
        return u

    @staticmethod
    def box_iou(a, b):
        u = MultiDetectionLoss.box_union(a, b)
        i = MultiDetectionLoss.box_intersection(a, b)
        iou = tensor.switch(tensor.eq(u, 0), 0, i/u)
        return iou

    @staticmethod
    def box_mse(a, b):
        mse = tensor.sum(tensor.square(a-b), axis=2)
        return mse

    @staticmethod
    def function(y_true, y_pred):
        loss = 0.0

        shape = (50, side * side, len(classes) + boxes * 5)

        offset = []
        for i in range(7):
            for j in range(7):
                offset.append(j)
                offset.append(i)
                offset.extend([0] * 3)
                offset.append(j)
                offset.append(i)
                offset.extend([0] * 6)

        offset = numpy.asarray(offset)
        y_pred_offset = y_pred + offset
        y_true_offset = y_true + offset

        y_pred = y_pred.reshape(shape)
        y_true = y_true.reshape(shape)
        y_pred_offset = y_pred_offset.reshape(shape)
        y_true_offset = y_true_offset.reshape(shape)

        a_offset = y_pred_offset[:, :, 0:4]
        b_offset = y_pred_offset[:, :, 5:9]
        gt_offset = y_true_offset[:, :, 0:4]

        a = y_pred[:, :, 0:4]
        b = y_pred[:, :, 5:9]
        gt = y_true[:, :, 0:4]

        iou_a_gt = MultiDetectionLoss.box_iou(a_offset, gt_offset)
        iou_a_gt = gradient.disconnected_grad(iou_a_gt)

        iou_b_gt = MultiDetectionLoss.box_iou(b_offset, gt_offset)
        iou_b_gt = gradient.disconnected_grad(iou_b_gt)

        mse_a_gt = MultiDetectionLoss.box_mse(a_offset, gt_offset)
        mse_b_gt = MultiDetectionLoss.box_mse(b_offset, gt_offset)

        mask = tensor.switch(tensor.lt(iou_a_gt, iou_b_gt), 1, 0)

        mask_iou_zero = tensor.switch(tensor.and_(tensor.le(iou_a_gt, 0), tensor.le(iou_b_gt, 0)), 1, 0)
        mask_mse = tensor.switch(tensor.lt(mse_a_gt, mse_b_gt), 1, 0)
        mask_change = mask_iou_zero * mask_mse
        mask = mask + mask_change
        mask = gradient.disconnected_grad(mask)

        loss_a_gt = tensor.sum(tensor.square(a - gt), axis=2) * 5
        loss_b_gt = tensor.sum(tensor.square(b - gt), axis=2) * 5

        loss += y_true[:, :, 4] * (1 - mask) * loss_a_gt
        loss += y_true[:, :, 4] * mask * loss_b_gt

        closs_a_gt = tensor.square(iou_a_gt * y_true[:, :, 4] - y_pred[:, :, 4])
        closs_b_gt = tensor.square(iou_b_gt * y_true[:, :, 4] - y_pred[:, :, 9])

        loss += closs_a_gt * (1 - mask) * y_true[:, :, 4]
        loss += closs_b_gt * mask * y_true[:, :, 4]

        loss += closs_a_gt * (1 - y_true[:, :, 4]) * 0.5
        loss += closs_b_gt * (1 - y_true[:, :, 4]) * 0.5

        loss += tensor.sum(tensor.square(y_pred[:, :, 10:13] - y_true[:, :, 10:13]), axis=2) * y_true[:, :, 4]

        loss = tensor.sum(loss)

        return loss


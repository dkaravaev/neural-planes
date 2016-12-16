import numpy

from theano import tensor, gradient

side = 7
boxes = 2
classes = ["fighter", "civil-plane", "bird"]


class SingleDetectionMetrics:
    """
    INPUT FORMAT:
        y_true = Tensor, dim(y_true) = (batch, side * side * (classes + obj + x + y + w + h))
        y_pred ~ y_true
    """
    PENALTY_OBJ = 4

    @staticmethod
    def function(y_true, y_pred):
        loss = 0.0

        scale_vector = numpy.asarray([SingleDetectionMetrics.PENALTY_OBJ] * 4)
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

    @staticmethod
    def accuracy(y_true, y_pred):
        acc = 0.0

        for i in range(49):
            acc += tensor.eq(y_true[:, i * 8:i * 8 + 4], y_pred[:, i * 8:i * 8 + 4])
            acc += tensor.eq(y_true[:, i * 8 + 4:i * 8 + 8], tensor.round(y_pred[:, i * 8 + 4:i * 8 + 8]))

        acc = tensor.sum(acc) / 19600
        return acc

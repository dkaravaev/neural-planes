from theano import tensor as T


def overlap(x1, w1, x2, w2, side):
    x1 /= side
    x2 /= side
    l1 = x1 - w1*w1/2
    l2 = x2 - w2*w2/2
    left = T.switch(T.lt(l1, l2), l2, l1)
    r1 = x1 + w1*w1/2
    r2 = x2 + w2*w2/2
    right = T.switch(T.lt(r1, r2), r1, r2)
    return right - left


def box_intersection(a, b):
    w = overlap(a[:, :, 0], a[:, :, 3], b[:, :, 0], b[:, :, 3], 7)
    h = overlap(a[:, :, 1], a[:, :, 2], b[:, :, 1], b[:, :, 2], 7)
    w = T.switch(T.lt(w, 0), 0, w)
    h = T.switch(T.lt(h, 0), 0, h)
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    area_a = a[:, :, 2] * a[:, :, 3]
    area_b = b[:, :, 2] * b[:, :, 3]
    u = area_a * area_a + area_b * area_b - i
    return u


def box_iou(a, b):
    u = box_union(a, b)
    i = box_intersection(a, b)
    iou = T.switch(T.eq(u, 0), 0, i/u)
    return iou


def box_mse(a, b):
    mse = T.sum(T.square(a-b), axis=2)
    return mse

import numpy
import math

from PIL import ImageDraw


class Box:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = -1
        self.prob = 0

    def __str__(self):
        return '%f %f %f %f %d %f' % (self.x, self.y, self.h, self.w, self.class_num, self.prob)


class DetectionConverter:
    def __init__(self, side, w, h, b):
        self.side = side
        self.w = w
        self.h = h
        self.b = b

        self.cell_size = self.w / self.side

        self.classes = ['fighter', 'civil-plane', 'bird']

    def convert_output(self, predictions, threshold):
        predictions = predictions.reshape(self.side, self.side, self.b * 5 + len(self.classes))

        boxes = []
        for row in range(self.side):
            for col in range(self.side):
                probs = predictions[row, col, 7] * predictions[row, col, 4:7]
                prob = numpy.max(probs)

                if prob >= threshold:
                    box = Box()
                    box.prob = prob
                    box.x = self.cell_size * (predictions[row, col, 0] + col)
                    box.y = self.cell_size * (predictions[row, col, 1] + row)

                    box.w = 224 * (math.pow(predictions[row, col, 2], 2))
                    box.h = 224 * (math.pow(predictions[row, col, 3], 2))

                    box.class_num = numpy.argmax(probs)

                    boxes.append(box)

        return boxes

    def overlay_results(self, image, predictions, result_path, threshold):
        boxes = self.convert_output(predictions, threshold)

        draw = ImageDraw.Draw(image)

        colors = ['red', 'green', 'blue']

        for box in boxes:
            print('Found {0} with P = {1} at (x = {2}, y = {3}, w = {4}, h ={5})'.format(self.classes[box.class_num],
                                                                                         box.prob, box.x, box.y, box.w,
                                                                                         box.h))

            xmin = box.x - box.w / 2
            xmax = box.x + box.w / 2
            ymin = box.y - box.h / 2
            ymax = box.y + box.h / 2
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=colors[box.class_num])

        image.save(result_path)

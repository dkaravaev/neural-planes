import numpy
import math

from PIL import Image
from PIL import ImageDraw


class Box:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = -1


class DetectionHandler:
    def __init__(self):
        self.side = 7
        self.w = 224
        self.h = 224
        self.b = 1

        self.cell_size = self.w / self.side

        self.classes = ['fighter', 'civil-plane', 'bird']

    def convert_output(self, output, threshold):
        output = output.reshape(self.side, self.side, self.b * 5 + len(self.classes))

        boxes = []
        for row in range(self.side):
            for col in range(self.side):
                probs = output[row, col, 4] * output[row, col, 5:8]
                prob = numpy.max(probs)

                if prob >= threshold:
                    box = Box(len(self.classes))

                    box.x = self.cell_size * (output[row, col, 0] + col)
                    box.y = self.cell_size * (output[row, col, 1] + row)
                    box.w = self.w * (math.pow(output[row, col, 2], 2))
                    box.h = self.h * (math.pow(output[row, col, 3], 2))

                    box.class_num = numpy.argmax(probs)
                    boxes.append(box)

        return boxes

    def overlay_results(self, image, output, threshold):
        boxes = self.convert_output(output, threshold)

        draw = ImageDraw.Draw(image)

        colors = ['red', 'green', 'blue']

        for box in boxes:
            color = colors.index(box.class_num)
            xmin = box.x - box.w / 2
            xmax = box.x + box.w / 2
            ymin = box.y - box.h / 2
            ymax = box.y + box.h / 2
            draw.rectangle(((xmin, xmax), (ymin, ymax)), outline=color)

        image.save('result.png')

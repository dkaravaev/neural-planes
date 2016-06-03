import numpy
import xml.etree.ElementTree as ET
import grid

from PIL import Image


class GridImage:
    def __init__(self, img, xml, side, w, h, b, classes):
        self.array = numpy.asarray(Image.open(img))
        self.array = numpy.asarray([self.array[:, :, 0] / 255, self.array[:, :, 1] / 255, self.array[:, :, 2] / 255],
                                   dtype='float32')

        self.side = side
        self.w = w
        self.h = h
        self.b = b
        self.classes = classes

        self.grid = grid.Grid(self.side, self.w, self.h)

        tree = ET.parse(xml)
        root = tree.getroot()
        for obj in root.iter('object'):
            geometry = obj.find('geometry')
            box = geometry.find('bndbox')

            xmin, ymin = float(box.find('xmin').text), float(box.find('ymin').text)
            xmax, ymax = float(box.find('xmax').text), float(box.find('ymax').text)
            class_num = self.classes.index(obj.find('name').text)

            self.grid.insert(box=grid.Box(xmin, ymin, xmax, ymax, class_num))

    def ground_truth(self):
        classes_num = len(self.classes)
        truth = numpy.zeros(self.side * self.side * (self.b * 5 + classes_num), dtype='float32')
        for i in range(self.side * self.side):
            col = i % self.side
            row = i / self.side

            boxes = self.grid.get(col, row)

            class_index = i * classes_num
            truth[class_index + boxes[0].class_num] = 1

            confidence_index = self.side * self.side * classes_num + i * self.b
            truth[confidence_index] = 1

            box_index = self.side * self.side * (classes_num + self.b) + i * self.b * 4
            x, y, w, h = boxes[0].normalize(self.side, self.w, self.h)
            truth[box_index + 0] = x
            truth[box_index + 1] = y
            truth[box_index + 2] = numpy.sqrt(w)
            truth[box_index + 3] = numpy.sqrt(h)

        return truth

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

    def truth_tensor(self):
        classes_num = len(self.classes)

        pred_dim = self.b * 5 + classes_num
        shape = (self.side, self.side, pred_dim)

        truth = numpy.zeros(shape, dtype='float32')

        for row in range(self.side):
            for col in range(self.side):
                boxes = self.grid.get(row, col)
                if not (boxes is None):
                    offset = 0
                    for box in boxes:
                        x, y, w, h = box.normalize(self.side, self.w, self.h)
                        truth[row, col, offset + 0] = x
                        truth[row, col, offset + 1] = y
                        truth[row, col, offset + 2] = numpy.sqrt(w)
                        truth[row, col, offset + 3] = numpy.sqrt(h)

                        offset += 4

                    # All boxes have one class num
                    truth[row, col, self.b * 4 + boxes[0].class_num] = 1
                    truth[row, col, self.b * 4 + classes_num] = 1

        return truth.reshape(shape[0] * shape[1] * shape[2])

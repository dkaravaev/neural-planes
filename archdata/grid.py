class Box:
    def __init__(self, xmin, ymin, xmax, ymax, class_num):
        self.x = (xmin + xmax) / 2.0
        self.y = (ymin + ymax) / 2.0
        self.w = xmax - xmin
        self.h = ymax - ymin
        self.class_num = class_num

    def normalize(self, side, w, h):
        return (self.x % side) / w, \
               (self.y % side) / h,  \
               self.w / w,  \
               self.h / h


class Cell:
    def __init__(self):
        self.boxes = []
        self.has_obj = False

    def add_box(self, box):
        self.boxes.append(box)
        self.has_obj = True


class Grid:
    def __init__(self, side, w, h):
        self.side = side
        self.w = w
        self.h = h
        self.cell_size = w / side
        self.cells = [[Cell()] * side] * side

    def insert(self, box):
        col = box.x // self.cell_size
        row = box.y // self.cell_size

        self.cells[int(row)][int(col)].add_box(box)

    def get(self, col, row):
        return self.cells[int(row)][int(col)].boxes


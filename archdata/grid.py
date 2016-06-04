class Box:
    def __init__(self, xmin, ymin, xmax, ymax, class_num):
        self.x = (xmin + xmax) / 2.0
        self.y = (ymin + ymax) / 2.0
        self.w = xmax - xmin
        self.h = ymax - ymin
        self.class_num = class_num

    def normalize(self, side, w, h):
        cell_size = int(w / side)
        return (self.x % cell_size) / cell_size, \
               (self.y % cell_size) / cell_size,  \
               self.w / w, self.h / h


class Cell:
    def __init__(self):
        self.boxes = list()
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

        self.cells = list()
        for i in range(side):
            self.cells.append(list())
            for j in range(side):
                self.cells[i].append(Cell())

        self.size = 0

    def insert(self, box):
        row = box.y // self.cell_size
        col = box.x // self.cell_size

        self.size += 1
        self.cells[int(row)][int(col)].add_box(box)

    def get(self, row, col):
        cell = self.cells[row][col]
        if cell.has_obj:
            return cell.boxes
        return None


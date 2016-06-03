import h5py
import numpy


class DataLoader:
    def __init__(self, filename):
        self.file = h5py.File(filename, 'r')
        numpy.random.seed(0)

    def flow(self, batch):
        while True:
            for offset in range(0, self.size(), batch):
                yield self.read_chunk(batch, offset)

    def read_chunk(self, chunk, offset):
        x = self.file['x_train'][offset:offset + chunk]
        y = self.file['y_train'][offset:offset + chunk]
        numpy.random.shuffle(x)
        numpy.random.shuffle(y)
        return x, y

    def size(self):
        return self.file['x_train'].shape[0]

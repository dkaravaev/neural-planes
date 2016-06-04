import h5py
import numpy


class DataLoader:
    def __init__(self, filename, size=None):
        self.file = h5py.File(filename, 'r')
        if size is None:
            self.size = self.file['x'].shape[0]
        else:
            self.size = size

        numpy.random.seed()

    def flow(self, batch):
        while True:
            offset = numpy.random.randint(0, self.size - batch)
            yield self.read_chunk(offset, batch)

    def read_chunk(self, offset, chunk):
        x = self.file['x'][offset:offset + chunk]
        y = self.file['y'][offset:offset + chunk]
        numpy.random.shuffle(x)
        numpy.random.shuffle(y)
        return x, y


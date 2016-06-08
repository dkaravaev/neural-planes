import h5py
import numpy
import pandas


class DataLoader:
    def __init__(self, filename, random=True, size=None):
        self.file = h5py.File(filename, 'r')
        self.random = random

        if size is None:
            self.size = self.file['x'].shape[0]
        else:
            self.size = size

        if random:
            self.series = pandas.Series(range(0, self.size))

        self.offset = 0
        numpy.random.seed(1234)

    def flow(self, batch):
        while True:
            if self.random:
                yield self.read_from_indexes(self.series.sample(batch))
            else:
                yield self.read_chunk(batch)

    def read_from_indexes(self, indexes):
        x = [self.file['x'][i] for i in indexes]
        y = [self.file['y'][i] for i in indexes]

        return numpy.asarray(x), numpy.asarray(y)

    def read_chunk(self, batch):
        x = self.file['x'][self.offset:self.offset + batch]
        y = self.file['y'][self.offset:self.offset + batch]

        self.offset += batch

        return x, y


import numpy

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from data_loader import DataLoader

"""
YOLO Input: Image in RGB
YOLO Output:
    SIDE x SIDE x CLASSES:
        P_{ijk}
        - Probability of ij-cell has object with k-class
    SIDE x SIDE x B:
        scale_{ij0} ... scale_{ijB}
        - Class scales for each bounding box in ij-cell
    SIDE x SIDE x B x 4:
        (x_{ij0}, y_{ij0}, sqrt(h_{ij0}), sqrt(w_{ij0})) ... (x_{ijB}, y_{ijB}, sqrt(h_{ijB}), sqrt(w_{ijB}))
        - Bounding box definition in each ij-cell
"""

"""
print(x_train[0].shape)
model.add(Convolution2D(512, 3, 3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(1024, 3, 3))
model.add(LeakyReLU(alpha=0.1))

model.add(Convolution2D(1024, 3, 3))
model.add(LeakyReLU(alpha=0.1))


model.add(Convolution2D(512, 3, 3))
model.add(LeakyReLU(alpha=0.1))

img_folder = '/home/dmitry/data/neural-planes/images'
xml_folder = '/home/dmitry/data/neural-planes/annotations'
x_train, y_train = list(), list()
for i in range(1000):
    x_train.append(numpy.random.uniform(size=shape))
    y_train.append(numpy.random.uniform(size=637))

model.fit(numpy.asarray(x_train), numpy.asarray(y_train), batch_size=32, nb_epoch=5)
"""

# TODO: MAKE VALIDATION GENERATOR!
# TODO: MAKE TEST DATA!
# TODO: MAKE LOSS FUNCTIONS!
# TODO: CONVERT PREDICTIONS!
# TODO: MAKE SINGLE CONFIG!
# TODO: MODEL WEIGHTS DIRECTORY!
# TODO: WHY WITH RANDOM DATA ALL WORKS???


shape = (3, 224, 224)

model = Sequential()

model.add(Convolution2D(16, 3, 3, input_shape=shape))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(128, 3, 3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(256, 3, 3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(512, 3, 3))
model.add(LeakyReLU(alpha=0.1))

model.add(Convolution2D(512, 3, 3))
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(.5))
model.add(Dense(49 * (5 * 2 + 3)))

sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')

loader = DataLoader('/home/dmitry/data/neural-planes/neural-planes.hdf5')
history = model.fit_generator(loader.train_flow(chunk=50), samples_per_epoch=10000, nb_epoch=10)
print(history.history)
model.save_weights('my_model_weights.h5')




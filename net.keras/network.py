import json
import datetime
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from loader import DataLoader


# TODO: CHECK CORRECTNESS OF DATA FLOW
# TODO: MAKE LOSS FUNCTIONS!!!!
# TODO: CONVERT PREDICTIONS!
# TODO: WHY WITH RANDOM DATA ALL WORKS???
# TODO: MAKE BETTER RANDOM IN LOADER!

class Network:
    """
    Model Input: Image in RGB
    Model Output:
        SIDE x SIDE x CLASSES:
            P_{ijk}
            - Probability of ij-cell has object with k-class
        SIDE x SIDE x B:
            scale_{ij0} ... scale_{ijB}
            - Class scales for each bounding box in ij-cell
        SIDE x SIDE x B x 4:
            (x_{ij0}, y_{ij0}, sqrt(h_{ij0}), sqrt(w_{ij0})) ... (x_{ijB}, y_{ijB}, sqrt(h_{ijB}), sqrt(w_{ijB}))
            - Bounding box definition in each ij-cell
        TOTAL = SIDE x SIDE x CLASSES + SIDE x SIDE x B + SIDE x SIDE x B x 4 = SIDE x SIDE x (B x 5 + CLASSES)
    """
    def __init__(self, filename):
        print('Config file: ' + filename)
        self.config = json.load(open(filename, 'r'))

        self.classes = self.config['global']['model']['classes']

        self.shape = (self.config['global']['image']['channels'],
                      self.config['global']['image']['size'][0],
                      self.config['global']['image']['size'][1])

        self.output = (self.config['global']['model']['side'] ** 2) * \
                      (self.config['global']['model']['boxes'] * 5 + len(self.classes))

        self.lr = self.config['net']['SGD']['rate']
        self.decay = self.config['net']['SGD']['decay']
        self.momentum = self.config['net']['SGD']['momentum']
        self.nesterov = self.config['net']['SGD']['nesterov']

        self.batch = self.config['net']['training']['batch']
        self.samples = self.config['net']['training']['samples']
        self.epochs = self.config['net']['training']['epochs']

        self.model = Sequential()
        self.form()

    def form(self):
        self.model.add(Convolution2D(16, 3, 3, input_shape=self.shape))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Convolution2D(128, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Flatten())

        self.model.add(Dense(256))
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(.5))

        self.model.add(Dense(self.output))

        sgd = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    def train(self):
        train_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
                                               self.config['global']['files']['datasets']['train']))

        validation_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
                                                    self.config['global']['files']['datasets']['train']))

        stopping = EarlyStopping(monitor='val_loss', patience=4)

        h = self.model.fit_generator(train_loader.flow(self.batch), validation_data=validation_loader.flow(self.batch),
                                     samples_per_epoch=self.samples, nb_epoch=self.epochs,
                                     nb_val_samples=validation_loader.size(), callbacks=[stopping])

        now = str(datetime.datetime.now())
        self.model.save_weights(os.path.join(self.config['global']['folders']['weights'],
                                             'weights_' + now + '.h5'))

        with open(os.path.join(self.config['global']['files']['histories'], 'history' + now + '.json'), 'w') as f:
            json.dump(obj=h.history, fp=f, indent=4)

        return h.history

    def test(self):
        test_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
                                              self.config['global']['files']['datasets']['test']))
        self.model.evaluate_generator(generator=test_loader.flow(batch=self.batch), val_samples=test_loader.size())



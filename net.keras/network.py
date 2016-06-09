import json
import datetime
import os
import numpy

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from objectives import SingleDetectionMetrics
from loader import DataLoader
from converter import DetectionConverter

# TODO: CHECK CORRECTNESS OF DATA FLOW
# TODO: MAKE LOSS FUNCTIONS!!!!
# TODO: CONVERT PREDICTIONS!
# TODO: WHY WITH RANDOM DATA ALL WORKS???
# TODO: MAKE BETTER RANDOM IN LOADER!


class Network:
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
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Dropout(.5))

        self.model.add(Dense(self.output, activation='sigmoid'))

        sgd = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov)
        self.model.compile(optimizer=sgd, loss=SingleDetectionMetrics.function,
                           metrics=[SingleDetectionMetrics.accuracy])

        weights_file = self.config['net']['weights']
        if not (weights_file is None):
            self.model.load_weights(weights_file)

    def train(self):
        train_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
                                               self.config['global']['files']['datasets']['train']))

        # validation_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
        #                                             self.config['global']['files']['datasets']['train']))

        # stopping = EarlyStopping(monitor='val_loss', patience=10)

        h = self.model.fit_generator(train_loader.flow(self.batch), samples_per_epoch=self.samples,
                                     nb_epoch=self.epochs)

        self.dump(h.history)

    def test(self):
        test_loader = DataLoader(os.path.join(self.config['global']['folders']['datasets'],
                                              self.config['global']['files']['datasets']['test']))
        self.model.evaluate_generator(generator=test_loader.flow(batch=self.batch), val_samples=test_loader.size)

    def dump(self, history):
        now = str(datetime.datetime.now()).replace(' ', '_')

        dump_folder = os.path.join(self.config['global']['folders']['dumps'], 'dump_' + now)
        os.makedirs(dump_folder)

        self.model.save_weights(os.path.join(dump_folder, 'weights_' + now + '.h5'))
        json_model = self.model.to_json()

        with open(os.path.join(dump_folder, 'model_' + now + '.json'), 'w') as f:
            f.write(json_model)

        with open(os.path.join(dump_folder, 'history_' + now + '.json'), 'w') as f:
            json.dump(obj=history, fp=f, indent=4)

        with open(os.path.join(dump_folder, 'config_' + now + '.json'), 'w') as f:
            json.dump(obj=self.config, fp=f, indent=4)

        os.system('yandex-disk sync --dir=~/Yandex.Disk')

    def predict(self, image_path, result_path, threshold=0.5):
        image = Image.open(image_path)
        inp = numpy.asarray(image) / 255
        inp = numpy.asarray([inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]])

        output = self.model.predict(numpy.asarray([inp]), batch_size=1)
        dhandler = DetectionConverter()

        dhandler.overlay_results(image, output[0], result_path=result_path, threshold=threshold)

    def from_darknet_weights(self):
        return self.shape








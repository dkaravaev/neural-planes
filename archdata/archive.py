import os
import h5py
import json

from image import GridImage


class DataArchiver:
    def __init__(self, filename):
        self.config = json.load(open(filename, 'r'))
        self.filename = filename

        self.images_folders = self.config['global']['folders']['images']
        self.annotations_folders = self.config['global']['folders']['annotations']

        self.w, self.h = self.config['global']['image']['size'][0], self.config['global']['image']['size'][1]
        self.side = self.config['global']['model']['side']
        self.b = self.config['global']['model']['boxes']
        self.channels = self.config['global']['image']['channels']
        self.classes = self.config['global']['model']['classes']

    def run(self, batch=1000, train=True, val=True, test=True):
        print('Config file: ' + self.filename)

        if train:
            print('Archiving training data:')
            result_filename = os.path.join(self.config['global']['folders']['datasets'],
                                           self.config['global']['files']['datasets']['train'])
            self.archive_from_folders(self.images_folders['train'], self.annotations_folders['train'],
                                      result_filename, batch)

        if val:
            print('Archiving validation data:')
            result_filename = os.path.join(self.config['global']['folders']['datasets'],
                                           self.config['global']['files']['datasets']['validation'])
            self.archive_from_folders(self.images_folders['validation'], self.annotations_folders['validation'],
                                      result_filename, batch)

        if test:
            print('Archiving test data:')
            result_filename = os.path.join(self.config['global']['folders']['datasets'],
                                           self.config['global']['files']['datasets']['test'])
            self.archive_from_folders(self.images_folders['test'], self.annotations_folders['test'],
                                      result_filename, batch)

        print('Done!')

    def archive_from_folders(self, img_folder, ann_folder, result_filename, batch):
        images = os.listdir(img_folder)
        file = h5py.File(result_filename, 'w')

        output_size = self.side * self.side * (self.b * 5 + len(self.classes))

        x = file.create_dataset('x', (len(images), self.channels, self.w, self.h),
                                dtype='float32', chunks=True)
        y = file.create_dataset('y', (len(images), output_size),
                                dtype='float32', chunks=True)

        i = 0
        number = len(images)
        for img in images:
            if (i + 1) % batch == 0:
                print('\tArchived: ' + str(i + 1) + '/' + str(number))

            img_file = os.path.join(img_folder, img)
            xml_file = os.path.join(ann_folder, img.replace('png', 'xml'))
            image = GridImage(img_file, xml_file, self.side, self.w, self.h, self.b, self.classes)

            x[i] = image.array
            y[i] = image.truth_tensor()
            i += 1

        print('\tSaved to: ' + result_filename)

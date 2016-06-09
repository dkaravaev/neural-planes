import random
import os
import json

from image import ImageHandler


class ImageGenerator:
    NUMBER_LEN = 7

    def __init__(self, filename):
        self.config_name = filename
        self.config = json.load(open(filename, 'r'))
        self.models = self.config['global']['files']['3dmodels']
        self.backgrounds = self.config['global']['files']['backgrounds']

        self.classmap = self.config['gendata']['3dmodel']['classmap']

        self.models_folder = self.config['global']['folders']['3dmodels']
        self.backgrounds_folder = self.config['global']['folders']['backgrounds']

        self.annotations_folders = self.config['global']['folders']['annotations']
        self.images_folders = self.config['global']['folders']['images']

        self.rotation_x = self.config['gendata']['3dmodel']['rotation']['x']
        self.rotation_y = self.config['gendata']['3dmodel']['rotation']['y']
        self.rotation_z = self.config['gendata']['3dmodel']['rotation']['z']

        self.size = self.config['gendata']['3dmodel']['size']
        self.blur = self.config['gendata']['effects']['blur']

        random.seed()

    def run(self, batch=1000, train=True, val=True, test=True):
        print('Config file: ' + self.config_name)

        if train:
            print('Generating training data:')
            self.generate_to_folders(self.images_folders['train'], self.annotations_folders['train'],
                                     self.config['gendata']['number']['train'], batch)

        if val:
            print('Generating validation data:')
            self.generate_to_folders(self.images_folders['validation'], self.annotations_folders['validation'],
                                     self.config['gendata']['number']['validation'], batch)

        if test:
            print('Generating test data:')
            self.generate_to_folders(self.images_folders['test'], self.annotations_folders['test'],
                                     self.config['gendata']['number']['test'], batch)

        print('Done!')

    def generate_to_folders(self, img_folder, xml_folder, number, batch):
        for i in range(0, number):
            name = str(i).zfill(self.NUMBER_LEN)
            if (i + 1) % batch == 0:
                print('\tGenerated: ' + str(i + 1) + '/' + str(number))

            model_index = random.randint(0, len(self.models) - 1)
            back_index = random.randint(0, len(self.backgrounds) - 1)

            model_file = os.path.join(self.models_folder, self.models[model_index])
            background_file = os.path.join(self.backgrounds_folder, self.backgrounds[back_index])
            image_file = os.path.join(img_folder, name + '.png')

            img = ImageHandler(model_file, background_file, image_file)

            size_x = random.randint(self.size[0], self.size[1])
            size_y = random.randint(self.size[0], self.size[1])

            img.overlaid_img(self.rotation_x, self.rotation_y, self.rotation_z,
                             (size_x, size_y), self.blur)
            object_class = self.classmap[self.models[model_index]]
            img.to_xml(os.path.join(xml_folder, name + '.xml'), object_class)


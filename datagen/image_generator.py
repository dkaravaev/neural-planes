import random

from image_handler import ImageHandler
from init_parser import InitParser


class ImageGenerator:
    NUMBER_LEN = 7

    def __init__(self, config_name):
        self.config_name = config_name
        self.config = InitParser(config_name)
        self.models = self.config.models()
        self.backgrounds = self.config.backgrounds()

        self.model_map = self.config.model_map()

        random.seed()

    def run(self, batch):
        print('Config file: ' + self.config_name)
        number = self.config.images_number()

        for i in range(0, number):
            name = str(i).zfill(self.NUMBER_LEN)
            if (i + 1) % batch == 0:
                print('Generated: ' + str(i + 1) + '/' + str(number))

            model_index = random.randint(0, len(self.models) - 1)
            back_index = random.randint(0, len(self.backgrounds) - 1)

            img = ImageHandler(self.config.models_folder() + self.models[model_index],
                               self.config.backgrounds_folder() + self.backgrounds[back_index],
                               self.config.imgs_folder() + name + '.png')

            size = self.config.size()
            size_x = random.randint(size[0], size[0])
            size_y = random.randint(size[0], size[0])

            img.overlaid_img(self.config.rotation_x(), self.config.rotation_y(), self.config.rotation_z(),
                             (size_x, size_y), self.config.noise(), self.config.blur())

            img.to_xml(self.config.annotations_folder() + name + '.xml', self.model_map)

        print('Done!')

generator = ImageGenerator('configs/sample.config')
generator.run(5)

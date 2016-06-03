import os
import h5py

from conf_parser import ConfParser
from grid_image import GridImage


class DataArchiver:
    def __init__(self, config_filename):
        self.config_filename = config_filename
        config = ConfParser(self.config_filename)
        self.result_filename = config.result_filename()
        self.file = h5py.File(self.result_filename, 'w')

        self.img_folder = config.imgs_folder()
        self.ann_folder = config.annotations_folder()

        self.w, self.h = config.size()[0], config.size()[1]
        self.side = config.side()
        self.b = config.boxes()
        self.channels = config.channels()
        self.classes = config.classes()

    def archive(self, batch):
        print('Config file: ' + self.config_filename)
        imgs = os.listdir(self.img_folder)
        output_size = self.side * self.side * (self.b * 5 + len(self.classes))

        x_train = self.file.create_dataset('x_train', (len(imgs), self.channels, self.w, self.h),
                                           dtype='float32', chunks=True)
        y_train = self.file.create_dataset('y_train', (len(imgs), output_size), dtype='float32', chunks=True)

        i = 0
        number = len(imgs)
        for img in imgs:
            if (i + 1) % batch == 0:
                print('Generated: ' + str(i + 1) + '/' + str(number))

            img_file = os.path.join(self.img_folder, img)
            xml_file = os.path.join(self.ann_folder, img.replace('png', 'xml'))
            image = GridImage(img_file, xml_file, self.side, self.w, self.h, self.b, self.classes)

            x_train[i] = image.array
            y_train[i] = image.ground_truth()
            i += 1

        print('Done!')
        print('Saved to :' + self.result_filename)

archiver = DataArchiver('../configs/archdata_534.config')
archiver.archive(1000)

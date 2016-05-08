import subprocess
import random
import numpy

from PIL import Image


def bounding_box(img):
    a = numpy.where(img != [255, 255, 255, 0])
    return (numpy.min(a[1]), numpy.max(a[1])), (numpy.min(a[0]), numpy.max(a[0]))


class ImageHandler:
    MODEL_RENDER = 'bin/model_render'

    def __init__(self, model_filename, background_filename, result_filename, mode='TOTAL'):
        random.seed()

        self.mode = mode

        self.model_filename = model_filename
        self.background_filename = background_filename
        self.result_filename = result_filename

    def trans_img(self, rotation_x, rotation_y, rotation_z, size):
        x = random.randint(rotation_x[0], rotation_x[1])
        y = random.randint(rotation_y[0], rotation_y[1])
        z = random.randint(rotation_z[0], rotation_z[1])

        subprocess.call(['./' + self.MODEL_RENDER, self.model_filename, self.result_filename,
                        str(x), str(y), str(z),
                        str(size[0]), str(size[1])],
                        stdout=subprocess.PIPE)

        img = self.make_transparent(Image.open(self.result_filename))
        img.save(self.result_filename, 'PNG')

        return img

    def overlaid_img(self, rotation_x, rotation_y, rotation_z, size, noise, blur, trans_img):
        if self.mode == 'TOTAL':
            img = self.trans_img(rotation_x, rotation_y, rotation_z, size)
            subprocess.call(['rm', self.result_filename])
        else:
            img = Image.open(trans_img)

        background = Image.open(self.background_filename)

        pos_x = random.randint(0, background.size[0] - size[0])
        pos_y = random.randint(0, background.size[1] - size[1])

        background.paste(img, (pos_x, pos_y), img)

        background.save(self.result_filename, 'PNG')

        subprocess.call(['mogrify', '+noise', noise, '-blur', str(blur), self.result_filename])

    @staticmethod
    def make_transparent(img):
        img = img.convert('RGBA')
        data = img.getdata()

        new_data = []
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        img.putdata(new_data)
        return img


handler = ImageHandler('/home/dmitry/Data/neural-planes/models/f14/F-14A_Tomcat.obj',
                       '/home/dmitry/Data/neural-planes/backgrounds/clearsky.jpg',
                       'sample.png')

handler.trans_img((0, 360), (0, 360), (0, 360), (120, 120))

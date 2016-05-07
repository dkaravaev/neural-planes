import subprocess
import random
from PIL import Image


class OverlayPlane:
    BKG_X = 448
    BKG_Y = 448

    MODEL_X = 120
    MODEL_Y = 120

    def __init__(self, model_filename, background_filename, result_filename):
        random.seed()

        self.model_filename = model_filename
        self.background_filename = background_filename
        self.result_filename = result_filename

    def render_model(self):
        path_prefix = '../model-render/'
        plane_filename = path_prefix + 'plane.png'

        a = random.uniform(-1.0, 1.0)
        b = random.uniform(-1.0, 1.0)
        c = random.uniform(-1.0, 1.0)
        d = random.uniform(-1.0, 1.0)

        quat = str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d)
        subprocess.call(['./' + path_prefix + 'model_render', self.model_filename, plane_filename, quat, "120", "120"],
                        stdout=subprocess.PIPE)

        img = Image.open(plane_filename)
        subprocess.call(['rm', plane_filename])

        return img

    def produce(self):
        img = self.render_model()
        img = img.convert('RGBA')
        data = img.getdata()

        new_data = []
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        img.putdata(new_data)

        pos_x = random.randrange(0, self.BKG_X - self.MODEL_X)
        pos_y = random.randrange(0, self.BKG_Y - self.MODEL_Y)

        background = Image.open(self.background_filename)
        background.paste(img, (pos_x, pos_y), img)
        background.save(self.result_filename, 'JPEG')

        subprocess.call(['mogrify', '+noise', 'Gaussian', '-blur', '20', self.result_filename])





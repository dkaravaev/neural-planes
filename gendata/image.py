import subprocess
import random

from lxml import etree
from PIL import Image, ImageOps


class ImageHandler:
    MODEL_RENDER = 'bin/model_render'

    def __init__(self, model_filename, background_filename, result_filename):
        random.seed()

        self.model_filename = model_filename
        self.background_filename = background_filename
        self.result_filename = result_filename

        self.x, self.y, self.z = 0, 0, 0
        self.bb = ()
        self.noise = ''
        self.blur = 0

    def trans_img(self, rotation_x, rotation_y, rotation_z, size):
        self.x = random.randint(rotation_x[0], rotation_x[1])
        self.y = random.randint(rotation_y[0], rotation_y[1])
        self.z = random.randint(rotation_z[0], rotation_z[1])

        subprocess.call(['./' + self.MODEL_RENDER, self.model_filename, self.result_filename,
                        str(self.x), str(self.y), str(self.z),
                        str(size[0]), str(size[1])],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

        img = Image.open(self.result_filename)
        bb = ImageOps.invert(img).getbbox()

        img = self.make_transparent(img.crop(bb))
        img.save(self.result_filename, 'PNG')

        return img

    def overlaid_img(self, rotation_x, rotation_y, rotation_z, size, blur):
        self.blur = blur

        img = self.trans_img(rotation_x, rotation_y, rotation_z, size)
        subprocess.call(['rm', self.result_filename])

        background = Image.open(self.background_filename)

        pos_x = random.randint(0, background.size[0] - img.size[0])
        pos_y = random.randint(0, background.size[1] - img.size[1])

        background.paste(img, (pos_x, pos_y), img)

        background.save(self.result_filename, 'PNG')

        # Python image magic interface
        subprocess.call(['mogrify', '-blur', str(blur), self.result_filename])

        self.bb = pos_x, pos_y, pos_x + img.size[0], pos_y + img.size[1]
        return

    def to_xml(self, filename, object_class):
        annotation = etree.Element('annotation')

        files = etree.SubElement(annotation, 'files')
        model = etree.SubElement(files, 'model')
        model.text = self.model_filename

        background = etree.SubElement(files, 'background')
        background.text = self.background_filename

        result = etree.SubElement(files, 'result')
        result.text = self.result_filename

        obj = etree.SubElement(annotation, 'object')
        name = etree.SubElement(obj, 'name')
        name.text = object_class

        geometry = etree.SubElement(obj, 'geometry')
        rotation = etree.SubElement(geometry, 'rotation')

        rotation_x = etree.SubElement(rotation, 'x')
        rotation_x.text = str(self.x)
        rotation_y = etree.SubElement(rotation, 'y')
        rotation_y.text = str(self.y)
        rotation_z = etree.SubElement(rotation, 'z')
        rotation_z.text = str(self.z)

        bndbox = etree.SubElement(geometry, 'bndbox')
        minx = etree.SubElement(bndbox, 'xmin')
        minx.text = str(self.bb[0])
        miny = etree.SubElement(bndbox, 'ymin')
        miny.text = str(self.bb[1])

        maxx = etree.SubElement(bndbox, 'xmax')
        maxx.text = str(self.bb[2])
        maxy = etree.SubElement(bndbox, 'ymax')
        maxy.text = str(self.bb[3])

        effects = etree.SubElement(obj, 'effects')
        blur = etree.SubElement(effects, 'blur')
        blur.text = str(self.blur)
        noise = etree.SubElement(effects, 'noise')
        noise.text = str(self.noise)

        with open(filename, 'wb') as f:
            f.write(etree.tostring(annotation,  pretty_print=True))

        f.close()
        return

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

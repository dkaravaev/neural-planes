import configparser


def parse_tuple(s):
    ss = s.split(' : ')
    return int(ss[0]), int(ss[1])


class InitParser:
    def __init__(self, filename):
        self.config = configparser.ConfigParser()
        self.config.read(filename)

    # IMAGES
    def images_number(self):
        return int(self.config['MAIN-INFO']['images_number'])

    # FOLDERS
    def backgrounds_folder(self):
        return self.config['FOLDERS']['backgrounds_folder']

    def models_folder(self):
        return self.config['FOLDERS']['models_folder']

    def imgs_folder(self):
        return self.config['FOLDERS']['images_folder']

    def annotations_folder(self):
        return self.config['FOLDERS']['annotations_folder']

    # FILES
    def backgrounds(self):
        return self.config['FILES']['backgrounds'].split(', ')

    def models(self):
        return self.config['FILES']['models'].split(', ')

    # MODEL-POSITION
    def size(self):
        return parse_tuple(self.config['MODEL-POSITION']['size'])

    def rotation_x(self):
        return parse_tuple(self.config['MODEL-POSITION']['rotation_x'])

    def rotation_y(self):
        return parse_tuple(self.config['MODEL-POSITION']['rotation_y'])

    def rotation_z(self):
        return parse_tuple(self.config['MODEL-POSITION']['rotation_z'])

    # EFFECTS
    def blur(self):
        return int(self.config['EFFECTS']['blur'])

    def noise(self):
        return self.config['EFFECTS']['noise']

    # CLASSES
    def model_map(self):
        kvs = [kv.split(' : ') for kv in self.config['CLASSES']['model_map'].split(', ')]
        return {self.models_folder() + kv[0]: kv[1] for kv in kvs}
import configparser


def parse_tuple(s):
    ss = s.split(' : ')
    return int(ss[0]), int(ss[1])


class ConfParser:
    def __init__(self, filename):
        self.config = configparser.ConfigParser()
        self.config.read(filename)

    def imgs_folder(self):
        return self.config['FOLDERS']['images_folder']

    def annotations_folder(self):
        return self.config['FOLDERS']['annotations_folder']

    def size(self):
        return parse_tuple(self.config['IMAGE']['size'])

    def channels(self):
        return int(self.config['IMAGE']['channels'])

    def boxes(self):
        return int(self.config['MODEL-PARAMS']['box_num'])

    def side(self):
        return int(self.config['MODEL-PARAMS']['side'])

    def classes(self):
        return self.config['MODEL-PARAMS']['classes'].strip().split(', ')

    def result_filename(self):
        return self.config['RESULT']['filename'].strip()
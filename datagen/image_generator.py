from image_handler import ImageHandler
from init_parser import InitParser


class ImageGenerator:
    def __init__(self, config_name):
        self.config = InitParser(config_name)

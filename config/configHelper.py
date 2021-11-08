import configparser
import os

configHelper = None
def get():
    global configHelper
    if configHelper is None: configHelper = ConfigHelper()
    return configHelper

class ConfigHelper:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.cfg'))
        self.thresholdValue = config.getfloat('Model', 'thresholdValue')
        self.trainImagePath = config.get('Model', 'trainImagePath')
        self.trainImageCollectionNo = config.getint('Model', 'trainImageCollectionNo')
        self.trainClassName = config.get('Model', 'trainClassName')
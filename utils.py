import numpy as np
import parsers
class HeadData:
    def __init__(self,data,indices):
        self.data = data
        self.indices = indices

    def split(self):
import numpy as np
from typing import Union, List
import os
import cv2


class Results:
    def __init__(self, engine):
        self.engine = engine

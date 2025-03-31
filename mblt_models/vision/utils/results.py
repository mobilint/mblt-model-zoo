import numpy as np
from typing import Union, List
import os
import cv2


class Results:
    def __init__(self, engine):
        self.pre_cfg = engine.pre_cfg
        self.post_cfg = engine.post_cfg

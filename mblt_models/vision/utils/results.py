import numpy as np
from typing import Union, List
import os
import cv2


class Results:
    def __init__(self, pre_cfg: dict, post_cfg: dict, output):
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg
        self.output = output

    @classmethod
    def from_engine(cls, engine, output):
        pre_cfg = engine.pre_cfg
        post_cfg = engine.post_cfg
        return cls(pre_cfg, post_cfg, output)

from .base import YOLOPostBase


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

    def rearrange(self, x):
        return x

    def decode(self, x):
        return x

    def nms(self, x):
        return x


class YOLOAnchorlessSegPost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

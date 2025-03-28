from abc import ABC, abstractmethod


class PostBase(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class YOLOPostBase(PostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__()
        img_size = pre_cfg.get("YoloPre")["img_size"]

        if isinstance(img_size, int):
            self.imh = self.imw = img_size
        elif isinstance(img_size, list):
            self.imh, self.imw = img_size

        self.nc = pre_cfg.get("nc")
        self.anchors = post_cfg.get("anchors", None)  # anchor coordinates
        if self.anchors is None:
            self.anchorless = True
            self.nl = post_cfg.get("nl")
            assert self.nl is not None, "nl should be provided for anchorless model"
        else:
            self.anchorless = False
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2

        self.n_extra = post_cfg.get("n_extra", 0)
        self.task = post_cfg.get("task")

    def __call__(self, x):
        x = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return x

    @abstractmethod
    def rearrange(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def nms(self, x):
        pass

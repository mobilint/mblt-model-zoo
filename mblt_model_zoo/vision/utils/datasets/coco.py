CLASS_INV_MAP = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
)

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

DET_PALETTE = [
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
    (255, 77, 255),
    (0, 226, 252),
    (182, 182, 255),
    (0, 82, 0),
    (120, 166, 157),
    (110, 76, 0),
    (174, 57, 255),
    (199, 100, 0),
    (72, 0, 118),
    (255, 179, 240),
    (0, 125, 92),
    (209, 0, 151),
    (188, 208, 182),
    (0, 220, 176),
    (255, 99, 164),
    (92, 0, 73),
    (133, 129, 255),
    (78, 180, 255),
    (0, 228, 0),
    (174, 255, 243),
    (45, 89, 255),
    (134, 134, 103),
    (145, 148, 174),
    (255, 208, 186),
    (197, 226, 255),
    (171, 134, 1),
    (109, 63, 54),
    (207, 138, 255),
    (151, 0, 95),
    (9, 80, 61),
    (84, 105, 51),
    (74, 65, 105),
    (166, 196, 102),
    (208, 195, 210),
    (255, 109, 65),
    (0, 143, 149),
    (179, 0, 194),
    (209, 99, 106),
    (5, 121, 0),
    (227, 255, 205),
    (147, 186, 208),
    (153, 69, 1),
    (3, 95, 161),
    (163, 255, 0),
    (119, 0, 170),
    (0, 182, 199),
    (0, 165, 120),
    (183, 130, 88),
    (95, 32, 0),
    (130, 114, 135),
    (110, 129, 133),
    (166, 74, 118),
    (219, 142, 185),
    (79, 210, 114),
    (178, 90, 62),
    (65, 70, 15),
    (127, 167, 115),
    (59, 105, 106),
    (142, 108, 45),
    (196, 172, 0),
    (95, 54, 80),
    (128, 76, 255),
    (201, 57, 1),
    (246, 0, 122),
    (191, 162, 208),
]

POSE_PALETTE = [
    [255, 128, 0],
    [255, 153, 51],
    [255, 178, 102],
    [230, 230, 0],
    [255, 153, 255],
    [153, 204, 255],
    [255, 102, 255],
    [255, 51, 255],
    [102, 178, 255],
    [51, 153, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 51, 51],
    [153, 255, 153],
    [102, 255, 102],
    [51, 255, 51],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 255],
]

POSE_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

LIMB_PALLETE = [
    POSE_PALETTE[i]
    for i in [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
]
KEYPOINT_PALLETE = [
    POSE_PALETTE[i] for i in [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
]


def get_coco_class_num() -> int:
    """Get the number of COCO classes.

    Returns:
        int: The number of COCO classes.
    """
    return len(CLASSES)


def get_coco_label(idx: int) -> str:
    """Get the COCO label by index.

    Args:
        idx (int): The index of the COCO label.

    Returns:
        str: The COCO label.
    """
    assert 0 <= idx < get_coco_class_num(), f"Invalid index: {idx}"

    return CLASSES[idx]


def get_coco_inv(idx: int) -> str:
    """Get the COCO label by index.

    Args:
        idx (int): The index of the COCO label.

    Returns:
        str: The COCO label.
    """
    return CLASS_INV_MAP[idx]


def get_coco_det_palette(idx: int) -> tuple:
    """Get the COCO detection palette by index.

    Args:
        idx (int): The index of the COCO detection palette.

    Returns:
        tuple: The COCO detection palette. (R, G, B)
    """
    return DET_PALETTE[idx]


def get_coco_pose_palette() -> tuple:
    """Get the COCO pose palette.

    Returns:
        tuple: The COCO pose palette.
    """
    return POSE_PALETTE


def get_coco_pose_skeleton() -> list:
    """Get the COCO pose skeleton.

    Returns:
        list: The COCO pose skeleton.
    """
    return POSE_SKELETON


def get_coco_limb_palette() -> list:
    """Get the COCO limb palette.

    Returns:
        list: The COCO limb palette.
    """
    return LIMB_PALLETE


def get_coco_keypoint_palette() -> list:
    """Get the COCO keypoint palette.

    Returns:
        list: The COCO keypoint palette.
    """
    return KEYPOINT_PALLETE

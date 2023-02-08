import numpy as np

ANCHORS_PER_GRID = 3
TOTAL_ANCHORS = 9

# YOLO_ANCHORS = [[[116, 90], [156, 198], [373, 326]],
#                 [[30,  61], [62,  45], [59,  119]],
#                 [[10,  13], [16,  30], [33, 23]]]

YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]

# YOLO_ANCHORS = [[[10,  20], [14.14,   14.14], [20,   10]],
#                 [[10,  20], [14.14,   14.14], [20,   10]],
#                 [[10,  20], [14.14,   14.14], [20,   10]]]

YOLO_STRIDES = [4, 8, 16]

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T

WIDTH = 416
HEIGHT = 416

NUM_CLASS = 10
MAX_BBOXES  = 100

SCORE_THRES = 0.4

OBJ_LOSS_WEIGHT = [4.0, 2.0, 1.0]

DATA_PATH = "/home/roboe/git/HSMNIST/data_yyminst_debug/dataset"
# DATA_PATH = "/home/roboe/git/HSMNIST/data_yyminst/dataset"
BEST_MODEL = "/home/roboe/git/HSMNIST/best.h5"
LAST_MODEL = "/home/roboe/git/HSMNIST/last.h5"
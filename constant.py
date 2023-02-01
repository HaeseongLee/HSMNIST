import numpy as np

ANCHORS_PER_GRID = 3
TOTAL_ANCHORS = 9

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

SCORE_THRES = 0.3

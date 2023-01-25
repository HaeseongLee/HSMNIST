import tensorflow as tf
import numpy as np
import math
import cv2
from constant import (STRIDES, ANCHORS, NUM_CLASS)

def bbox_ious(boxes1, boxes2, mode="iou"):
    '''
        Input format: (cx, cy, w, h)
            boxes1 : prediction
            boxes2 : label
    '''

    # convert (cx, cy, w, h) to (x_min, y_min, x_max, y_max)
    b1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    b2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    b1 = tf.concat([tf.minimum(b1[..., :2], b1[..., 2:]),
                        tf.maximum(b1[..., :2], b1[..., 2:])], axis=-1)
    b2 = tf.concat([tf.minimum(b2[..., :2], b2[..., 2:]),
                        tf.maximum(b2[..., :2], b2[..., 2:])], axis=-1)

    boxes1_area = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    boxes2_area = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    left_up = tf.maximum(b1[..., :2], b2[..., :2])
    right_down = tf.minimum(b1[..., 2:], b2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    esp = 1e-7 # for stable computation
    iou = inter_area / (union_area + esp)

    if mode == "iou":
        return iou
    
    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(b1[..., :2], b2[..., :2])
    enclose_right_down = tf.maximum(b1[..., 2:], b2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]
    
    if mode == "giou":
        # Calculate the GIoU value according to the GioU formula                   
        if np.max(enclose_area) == math.inf: # for computation stability 
            print("giou is divergent")            
            giou = -1.0
        else:
            giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + esp)

        return giou

    enclose_diag = enclose_right_down - enclose_left_up
    c = tf.math.reduce_euclidean_norm(enclose_diag, axis=-1)
    c2 = c**2  # length of diagonal from the enclose box.
    rho = boxes1[...,:2] - boxes2[...,:2]
    rho = tf.math.reduce_euclidean_norm(rho, axis=-1)
    rho2 = rho**2

    if mode == "diou":
        diou = iou - rho2/(c2 + esp)
        return diou

    if mode == "ciou":
        x = tf.math.atan(boxes2[...,2]/(boxes2[...,3]+esp)) - tf.math.atan(boxes1[...,2]/(boxes1[...,3]+esp))
        v = (4/math.pi)**2*tf.math.pow(x, 2)
        a = v / (1 - iou + v + esp)            
        ciou = iou - rho2/(c2 + esp) - a*v

        # print("ciou: ", tf.reduce_mean(ciou))

        m_ciou = tf.reduce_mean(ciou)
        if tf.math.is_nan(m_ciou):
            print("Nan!!")
        return ciou

def decode(x_in):
    # outs = self.model.outputs
    pred_tensors = []
    for i, out in enumerate(x_in):
        
        b = out.shape[0] # batch_size
        w = out.shape[1]
        h = out.shape[2]

        # reshape the output 
        conv = tf.reshape(out, (-1, w, h, 3, 5+NUM_CLASS))            
        raw_dxdy = conv[:, :, :, :, 0:2] # offset of center position     
        raw_dwdh = conv[:, :, :, :, 2:4] # Prediction box length and width offset
        raw_conf = conv[:, :, :, :, 4:5] # confidence of the prediction box
        raw_prob = conv[:, :, :, :, 5: ] # category probability of the prediction box 

        # next need Draw the grid. Where output_size is equal to 160*120, 80*60 or 40*30  
        y = tf.range(h, dtype=tf.int32)
        x = tf.range(w, dtype=tf.int32)
        xy = tf.meshgrid(x, y)
        xy = tf.transpose(xy, perm=[2,1,0]) # [width, height,2]
        xy = tf.tile(xy[:, :, tf.newaxis, :], [1, 1, 3, 1])
        xy_grid = tf.cast(xy, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(raw_dxdy) + xy_grid) * STRIDES[i]

        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_conf = tf.sigmoid(raw_conf) # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(raw_prob) # calculating the predicted probability category box object

        pred_tensor = tf.concat([pred_xy, pred_wh, pred_conf, pred_prob], axis=-1)
        pred_tensors.append(pred_tensor)

    return pred_tensors
    
def draw_bbox(image, bboxes):
    '''
    Argument:
        bboxes: [N, 6], (x_min, y_min, x_max, y_max, prob, id)
    '''

    bbox_color = (0, 0, 1)

    im = image[0,...].numpy()
        
    for i, bbox in enumerate(bboxes):                
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        id = int(bbox[5])
        # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(im, c1, c2, bbox_color, 2)

    return im

def xywh2yxyx(xywh):
    yxyx = np.zeros_like(xywh)
    yxyx[:,0] = xywh[:,1] - xywh[:,3]/2
    yxyx[:,1] = xywh[:,0] - xywh[:,2]/2
    yxyx[:,2] = xywh[:,1] + xywh[:,3]/2
    yxyx[:,3] = xywh[:,0] + xywh[:,2]/2
    
    return yxyx
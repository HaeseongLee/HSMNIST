import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.keras import Model

from architecture import HSMNIST
from load_data import DataLoader
from train import Trainer

from utils import decode, draw_bbox, xywh2yxyx
from constant import (STRIDES, ANCHORS, ANCHORS_PER_GRID,
                      WIDTH, HEIGHT, NUM_CLASS, SCORE_THRES)

class Detector:
    def __init__(self):
        pass

    def postprocess(self, preds):
        '''
        Arguments
            preds: (3, )
            - preds[i] = [N, W, H, 3, 5 + n_class], (cx, cy, w, h, p, class)

        Return
            pred: [M, 6], (x_min, y_min, x_max, y_max, conf, class_id)
                   different grid scales' outputs are resized as the original (e.g., 480x640)
        '''
        valid_scale = [0, np.inf]
        
        ori_h = HEIGHT
        ori_w = WIDTH
        
        tmp = []
        for i in range(ANCHORS_PER_GRID):
            pred = np.array(preds[i])
            tmp.append(pred)

        preds_arr = [np.reshape(x,(-1,np.shape(x)[-1])) for x in tmp]
        preds_arr = np.concatenate(preds_arr, axis=0)

        xywh = preds_arr[:, :4]
        conf = preds_arr[:, 4]
        prob = preds_arr[:, 5:]
        
        # print("prob:\n",prob)
        # print(np.argmax(prob, axis=-1))
        # print(np.max(prob))

        # (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
        coor = np.concatenate([xywh[:,:2]-xywh[:,2:]*0.5, xywh[:,:2]+xywh[:,2:]*0.5], axis=-1)

        # check invalid (x_min, y_min, x_max, y_max)
        clip_xy_min = np.maximum(coor[:,:2], [0,0])
        clip_xy_max = np.minimum(coor[:,2:], [ori_w-1, ori_h-1])
        coor = np.concatenate([clip_xy_min, clip_xy_max], axis=-1)
        invalid_mask = np.logical_or((coor[:,0] > coor[:,2]), (coor[:,1] > coor[:,3]))
        
        # check invalid bboxes 
        area = np.sqrt(np.multiply.reduce(coor[:, 2:4] - coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < area), (area < valid_scale[1]))

        # discard boxes with low scores
        classes = np.argmax(prob, axis=-1)
        scores = conf * prob[np.arange(len(coor)), classes]
        score_mask = scores > SCORE_THRES
        mask = np.logical_and(scale_mask, score_mask)
        
        coor, scores, classes = coor[mask], scores[mask], classes[mask]    

        return np.concatenate([coor, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)



if __name__ == "__main__":
    # model_path = "/home/roboe/roboe_ws/src/roboeod/script/RBDN/20230111_ciou/best.h5"
    model_path = "/home/roboe/git/HSMNIST/best.h5"

    # path = "/home/roboe/git/HSMNIST/data_yyminst/dataset"
    path = "/home/roboe/git/HSMNIST/data_yyminst_debug/dataset"

    dl = DataLoader(path)
    im, ld, pd, ns = dl.get_dataset()    

    batch_size = 1
    nb = math.ceil(ns/batch_size) # number of batch

    im = im.batch(batch_size)
    ld = ld.batch(batch_size)
    pd = pd.batch(batch_size)

    learning_info = {}
    learning_info["steps_per_epoch"] = nb
    learning_info['epochs'] = 1

    model = HSMNIST((WIDTH, HEIGHT, 3), NUM_CLASS).build(training=False)        
    model.load_weights(model_path)
    
    # print(model.layers[-1].weights)
        
    detector = Detector()
    for c, l, p in zip(im, ld, pd):    
        preds = model(c)
        preds = decode(preds)
        print(preds[0][0,:,:,0,5])
        cx = tf.range(0,400,10)
        cy = tf.range(0,400,10)
        
        im = c[0].numpy()
        # for i in range(ANCHORS_PER_GRID):
        #     x = cx/STRIDES[i]
        #     y = cy/STRIDES[i]
                
        #     x = np.array(x, dtype=np.int8)
        #     y = np.array(y, dtype=np.int8)
            
            
        #     for n in x:
                
        #         for m in y:
        #             im = c[0].numpy()
            
        #             xx = tf.reshape(preds[i][0, n, m,:,0], -1)
        #             yy = tf.reshape(preds[i][0, n, m,:,1], -1)
        #             ww = tf.reshape(preds[i][0, n, m,:,2], -1)
        #             hh = tf.reshape(preds[i][0, n, m,:,3], -1)  
            
        #             for j in range(3):
        #                 x_min = int(xx[j] - ww[j]/2)
        #                 y_min = int(yy[j] - hh[j]/2)
        #                 x_max = int(xx[j] + ww[j]/2)
        #                 y_max = int(yy[j] + hh[j]/2)
        #                 if j == 0: color=(0,0,255)
        #                 if j == 1: color=(0,255,0)
        #                 if j == 2: color=(255,0,0)
        #                 cv2.rectangle(im, (x_min,y_min), (x_max,y_max), color, 1)


        # # cv2.imshow("im", image_data[0].numpy())
        #             cv2.imshow("img", im)            
        #             cv2.waitKey(100)
        #             cv2.destroyAllWindows()
                
        
        result = detector.postprocess(preds)
        # print(result[0][0,0,0,0,:])
        bboxes = result[:,:4]
        scores = result[:,4]
        
        obj = preds[0][0,:,:,0,4].numpy()
        obj = cv2.resize(obj,(WIDTH, HEIGHT))
        cv2.imshow("obj", obj)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        prob = preds[0][0,:,:,0,5].numpy()
        print(np.max(preds[0][0,:,:,0,5]))
        prob = cv2.resize(prob, (WIDTH, HEIGHT))
        cv2.imshow("prob", prob)
        cv2.waitKey()
        cv2.destroyAllWindows()
        yxyx = xywh2yxyx(bboxes)

        best_bbox_indices = tf.image.non_max_suppression(yxyx, scores, 
                                                        max_output_size = 20,
                                                        iou_threshold = 0.5,
                                                        score_threshold = 0.5)
        best_bbox = result[best_bbox_indices,:]    
        # print(best_bbox)    
        img = draw_bbox(c, best_bbox)   
        # img = np.transpose(img, (1,0,2))
        im = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
        cv2.imshow("im", img.astype('float32'))
        cv2.waitKey()
        cv2.destroyAllWindows()
        break
import os
import random
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import cv2
import copy

from constant import (STRIDES, ANCHORS, ANCHORS_PER_GRID,
                      WIDTH, HEIGHT, MAX_BBOXES, NUM_CLASS)

class DataLoader:
    def __init__(self, path):
        self.path = path
    
    def _read_filename(self, path):
        files = glob.glob(os.path.join(path,"images/train/*"))        
        f_names = []
        for i in range(len(files)):
            f_names.append(os.path.splitext(os.path.split(files[i])[1])[0])
        return f_names

    def _resigter_filename(self, path):        
        files = self._read_filename(path)
        num_samples = len(files)
        mnist_dir = []        
        label_dir = []

        mnist_head = os.path.join(path,"images/train/")        
        label_haed = os.path.join(path,"labels/train/")
        
        for i in range(num_samples):
            mnist_dir.append(os.path.join(mnist_head,files[i]+".jpg"))            
            label_dir.append(os.path.join(label_haed,files[i]+".txt"))
        
        return mnist_dir, label_dir, num_samples

    def _get_color(self,fname):        
        color = tf.io.read_file(fname)
        color = tf.io.decode_image(color, channels=3)        
        color = tf.cast(color, tf.float32)/255
        return color
    
    def _get_label(self, fname):        
        label = np.loadtxt(fname.decode('utf-8'))
        if np.shape(label) == (5,):
            label = label.reshape(1,5)
        label_map, bboxes_xywh = self._process_label(label)
        
        info = []

        for i in range(len(label_map)):
            label_map[i] = label_map[i].astype(np.float32)
            info.append(label_map[i])

        for i in range(len(bboxes_xywh)):
            bboxes_xywh[i] = bboxes_xywh[i].astype(np.float32)
            info.append(bboxes_xywh[i])

        return info
 
    def _process_label(self, label_data):
        '''
            Get label maps to compute "bbox loss"
            Input: 
                label_data: (N, )
                            all labels from reading .txt file
            Output:
                label_map       :   (ANCHORS_PER_GRID, )                                
                label_map[i]    :   [img_width, img_height, ANCHORS_PER_GRID, 5 + num_class]
                bboxes_tensor:  (N, )
                                bboxes_tensor[i]: (ANCHORS_PER_GRID, )
                                bboxes_tensor[i][j] = [idx, cx, cy, w, h]

        '''
        w = WIDTH / STRIDES
        h = HEIGHT / STRIDES
                   
        label_map = [np.zeros((int(w[i]), int(h[i]), ANCHORS_PER_GRID, 5 + NUM_CLASS)) for i in range(3)]
        bboxes_xywh = [np.zeros((MAX_BBOXES, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        
        for bbox in label_data:              
            bbox_xywh = copy.deepcopy(bbox[1:])
            bbox_xywh[[0,2]] *= WIDTH
            bbox_xywh[[1,3]] *= HEIGHT
            
            bbox_class_index = int(bbox[0])
            onehot = np.zeros(NUM_CLASS, dtype=np.float)
            onehot[bbox_class_index] = 1.0
            uniform_distribution = np.full(NUM_CLASS, 1.0 / NUM_CLASS)
            delta = 0.001
            onehot = onehot * (1 - delta) + delta * uniform_distribution

            bbox_xywh_scaled = bbox_xywh / STRIDES[:, np.newaxis]
            iou = []
            exist_positive = False

            for i in range(ANCHORS_PER_GRID):            
                
                anch_xywh = np.zeros((ANCHORS_PER_GRID, 4))                
                anch_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5                
                anch_xywh[:, 2:4] = ANCHORS[i]

                a = self._xywh2yxyx(bbox_xywh_scaled[i,:]) 
                b = np.zeros((3,4)) # anchors
                for k in range(3):
                    b[k] = self._xywh2yxyx(anch_xywh[k,:])    
                
                score = np.array(tfa.losses.giou_loss(a, b, 'iou'))
                iou.append(score)
                iou_mask = score > 0.3
                
                # generate ground truth map
                if np.any(iou_mask):
                    xidx, yidx = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    y_min, x_min, y_max, x_max = np.round(a).astype(np.int32)
                    
                    label_map[i][yidx, xidx, iou_mask, :] = 0
                    label_map[i][yidx, xidx, iou_mask, 0:4] = bbox_xywh
                    label_map[i][y_min:y_max+1, x_min:x_max+1, iou_mask, 4:5] = 1.0
                    # label_map[i][yidx, xidx, iou_mask, 4:5] = score[iou_mask].reshape(-1,1)
                    #TODO: change "1.0" to one-hot for multiple labels
                    label_map[i][yidx, xidx, iou_mask, 5:] = onehot                 
                    bbox_ind = int(bbox_count[i] % MAX_BBOXES)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    
                    exist_positive = True
            
            # set the second best anchor if all ious has poor score
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / ANCHORS_PER_GRID) # 0 1 2
                best_anchor = int(best_anchor_ind % ANCHORS_PER_GRID)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label_map[best_detect][yind, xind, best_anchor, :] = 0
                label_map[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                # label_map[best_detect][yind, xidx, best_anchor, 4:5] = 1.0
                label_map[best_detect][y_min:y_max+1, x_min:x_max+1, best_anchor, 4:5] = 1.0
                # label_map[best_detect][yind, xind, best_anchor, 4:5] = iou[best_anchor_ind]
                label_map[best_detect][yind, xind, best_anchor, 5:] = onehot

                bbox_ind = int(bbox_count[best_detect] % MAX_BBOXES)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        
        return label_map, bboxes_xywh

    def _xywh2yxyx(self, xywh):
        yxyx = [xywh[1] - xywh[3]/2,
                xywh[0] - xywh[2]/2,
                xywh[1] + xywh[3]/2,
                xywh[0] + xywh[2]/2]
        return yxyx
    

    def get_dataset(self):
        mnists, ls, num_samples = self._resigter_filename(self.path) 

        mnist_path = tf.data.Dataset.from_tensor_slices(mnists)
        label_path = tf.data.Dataset.from_tensor_slices(ls)

        im = mnist_path.map(
            lambda x: tf.numpy_function(self._get_color, [x], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ld = label_path.map(
            lambda x: tf.numpy_function(self._get_label, [x], [tf.float32, tf.float32, tf.float32,
                                                               tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return im, ld, label_path, num_samples


if __name__=="__main__":
    path = "/home/roboe/git/HSMNIST/data_yyminst/dataset"

    dl = DataLoader(path)
    im, ld, pd, ns = dl.get_dataset()    

    cd = im.batch(1)
    ld = ld.batch(1)
    pd = pd.batch(1)

    for c, l in zip(cd, ld):
        im = c[0].numpy()        
        # im = np.transpose(im, (1,0,2))
        

        label_map, bboxes_xywh = l[0], l[3]

        xx = tf.reshape(label_map[0,:,:,0,0], -1)
        yy = tf.reshape(label_map[0,:,:,0,1], -1)
        ww = tf.reshape(label_map[0,:,:,0,2], -1)
        hh = tf.reshape(label_map[0,:,:,0,3], -1)
        valid_index = np.where(ww > 0.0)[0]
        
        x_map = label_map[0,:,:,0,0].numpy()
        x_map = cv2.resize(x_map, (416, 416))
        seg_mask = label_map[0,:,:,0,4].numpy()
        seg_mask = cv2.resize(seg_mask, (416, 416))

        y = cv2.addWeighted(im[:,:,0], 0.3, seg_mask, 0.8, 0.0)

        for i in valid_index:                
            x_min = int(xx[i] - ww[i]/2)
            y_min = int(yy[i] - hh[i]/2)
            x_max = int(xx[i] + ww[i]/2)
            y_max = int(yy[i] + hh[i]/2)            
            cv2.rectangle(y, (x_min,y_min), (x_max,y_max), (255,0,0), 1)

        # cv2.imshow("im", image_data[0].numpy())
        cv2.imshow("img", y)
        # cv2.imshow("seg", seg_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
        break
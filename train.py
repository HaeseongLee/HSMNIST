import tensorflow as tf
import numpy as np
import cv2
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from architecture import HSMNIST
from load_data import DataLoader

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


from history import LearningHistory
from scheduler import LearningScheduler

from utils import bbox_ious, decode
from constant import *

IOU_LOSS_THRESH = 0.5
WARMUP_EPOCHS = 1
LR_INIT =  1e-4
LR_END = 1e-6
EPOCHS = 10 #100


class Trainer:
    def __init__(self, model, info):
        self.model = model        
        self.spe = info["steps_per_epoch"]
        self.epochs = info["epochs"]

        # self.cs = tf.Variable(1, trainable=False, dtype=tf.int64) # cs : current steps
        self.cs = 0
        self.ws = WARMUP_EPOCHS * self.spe # ws : warmup_steps
        self.ts = self.epochs * self.spe  # ts : total_steps

        # self.input_size = model.input_shape[0][1:3]
        self.input_size = model.input_shape[1:3]

        # self.writer = tf.summary.create_file_writer(LOG_DIR)
        # self.input_size = model.input_shape[0][1:3] # height, width

        self.ave_losses = np.zeros(4)

        # lr = ExponentialDecay(initial_learning_rate=LR_INIT, decay_steps=1, decay_rate=0.9)
        # self.optimizer = Adam(learning_rate=LR_INIT)
        self.optimizer = SGD(learning_rate=LR_INIT, momentum=0.9)

        self.lh = LearningHistory(self.ts)
        self.ls = LearningScheduler(patience=30, save_setp=10)

        self.learning_stop = False 

        self.bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.cce = CategoricalCrossentropy()
        
    def compute_loss(self, pred, label_map, bboxes, w=1.0):
        ''' 
        Arguments
            pred: [N, W, H, ANCHORS_PER_GRID, 5 + n_class]
            
            label_map: [N, W, H, ANCHORS_PER_GRID, 5 + n_class]
                        grid map corresponds to the ground truth

            bboxes: [N, MAX_BOXES, 4], (cx, cy, w, h)
                    bbox information corresponds to the ground truth
        '''
        
        # input_w = np.shape(pred)[1]
        # input_h = np.shape(pred)[2]

        
        #TODO: compare pred_conf vs conv_raw_conf from the original code                
        pred_xywh = pred[:,:,:,:,0:4]
        pred_conf = pred[:,:,:,:,4:5]
        pred_prob = pred[:,:,:,:,5:]

        label_xywh = label_map[:,:,:,:,0:4]
        label_conf = label_map[:,:,:,:,4:5] # confidence for objectness
        label_prob = label_map[:,:,:,:,5:]
    
        
        # compute giou loss        
        giou = tf.expand_dims(bbox_ious(pred_xywh, label_xywh, "ciou"), axis=-1)  
        
        # bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_w * input_h)                
        # giou_loss = respond_bbox*bbox_loss_scale*(1-giou)
        # giou_loss = label_conf*(1-giou)
        giou_loss = (1-giou)


        iou = bbox_ious(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :], "iou")
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - label_conf) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )
        # respond_bgd = (1.0 - respond_bbox) # why max_iou is required?...
                
        
        alpha = 0.25  
        gamma = 2.0
        
        ce = self.bce(label_conf, pred_conf)[:,:,:,:,np.newaxis]
        p_t = (label_conf*pred_conf) + ((1 - label_conf) * (1 - pred_conf))
        
        alpha_factor = label_conf*alpha + (1-label_conf)*(1-alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        conf_loss = alpha_factor*modulating_factor*ce
                
                
        # prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=pred_prob)
        # prob_loss = conf_focal * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=pred_prob)
        prob_loss = self.cce(label_prob, pred_prob)
        # print("max conf: ", np.max(pred_prob), "\tmin conf: ", np.min(pred_prob))
        giou_loss = tf.reduce_mean(giou_loss)
        # conf_loss = tf.reduce_mean(conf_loss)
        conf_loss = tf.reduce_sum(conf_loss)
        prob_loss = tf.reduce_sum(prob_loss)

        giou_loss = 0.01*giou_loss
        conf_loss = w*conf_loss
        prob_loss = 0.00*prob_loss

        return giou_loss, conf_loss, prob_loss
    
    def train_step(self, image_data, target, epoch):        
        with tf.GradientTape() as tape:
            pred = self.model(image_data)       
            pred = decode(pred)     
            label_map, bboxes_xywh = target[0:3], target[3:6]

            # gt = label_map[2][0,:,:,0,4].numpy()
            # gt = cv2.resize(gt,(WIDTH,HEIGHT))
            p = pred[0][0,:,:,0,4].numpy()
            # tf.print("min: ", tf.reduce_min(p), "py max: ", tf.reduce_max(p))
            p = cv2.resize(p,(WIDTH,HEIGHT))
            # q = pred[0][0,:,:,0,5].numpy()
            # q = cv2.resize(q, (WIDTH, HEIGHT))
            
            # print(pred[0][0,:,:,0,5:])
            # print(np.max(q))
            # cv2.imshow("gt", gt)
            cv2.imshow("obj_map", p)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            
            # self.y = image_data[0].numpy()
            # y = cv2.addWeighted(image_data[0].numpy(), 0.3, respond_bgd, 5.0, 0.0)
            
            # cx = tf.reshape(label_map[0][0,:,:,0,0], -1)
            # cy = tf.reshape(label_map[0][0,:,:,0,1], -1)
            # ww = tf.reshape(label_map[0][0,:,:,0,2], -1)
            # hh = tf.reshape(label_map[0][0,:,:,0,3], -1)
            # valid_index = np.where(ww > 0.0)[0]
            
            # for i in valid_index:                
            #     x_min = int(cx[i] - ww[i]/2)
            #     y_min = int(cy[i] - hh[i]/2)
            #     x_max = int(cx[i] + ww[i]/2)
            #     y_max = int(cy[i] + hh[i]/2)
            #     cv2.rectangle(self.y, (x_min,y_min), (x_max,y_max), (0,0,0), 1)
            #     break
            
            
            # for i in range(ANCHORS_PER_GRID):
            #     x_target = int(cx[valid_index[0]]/STRIDES[i])
            #     y_target = int(cy[valid_index[0]]/STRIDES[i])
                
            #     xx = tf.reshape(pred[i][0,x_target,y_target,:,0], -1)
            #     yy = tf.reshape(pred[i][0,x_target,y_target,:,1], -1)
            #     ww = tf.reshape(pred[i][0,x_target,y_target,:,2], -1)
            #     hh = tf.reshape(pred[i][0,x_target,y_target,:,3], -1)  
                
            #     for j in range(3):
            #         x_min = int(xx[j] - ww[j]/2)
            #         y_min = int(yy[j] - hh[j]/2)
            #         x_max = int(xx[j] + ww[j]/2)
            #         y_max = int(yy[j] + hh[j]/2)
            #         if j == 0: color=(0,0,255)
            #         if j == 1: color=(0,255,0)
            #         if j == 2: color=(255,0,0)
            #         cv2.rectangle(self.y, (x_min,y_min), (x_max,y_max), color, 1)
            

            #     x_target = int(200/STRIDES[i])
            #     y_target = int(200/STRIDES[i])
                
            #     xx = tf.reshape(pred[i][0,x_target,y_target,:,0], -1)
            #     yy = tf.reshape(pred[i][0,x_target,y_target,:,1], -1)
            #     ww = tf.reshape(pred[i][0,x_target,y_target,:,2], -1)
            #     hh = tf.reshape(pred[i][0,x_target,y_target,:,3], -1)  
                
            #     for j in range(3):
            #         x_min = int(xx[j] - ww[j]/2)
            #         y_min = int(yy[j] - hh[j]/2)
            #         x_max = int(xx[j] + ww[j]/2)
            #         y_max = int(yy[j] + hh[j]/2)
            #         if j == 0: color=(0,0,255)
            #         if j == 1: color=(0,255,0)
            #         if j == 2: color=(255,0,0)
            #         cv2.rectangle(self.y, (x_min,y_min), (x_max,y_max), color, 1)


            # # cv2.imshow("im", image_data[0].numpy())
            # cv2.imshow("img", self.y)            
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            conf_loss_weight = [1.0, 1.0, 1.0]
            for i in range(ANCHORS_PER_GRID ):
                loss_items = self.compute_loss(pred[i], label_map[i], bboxes_xywh[i], OBJ_LOSS_WEIGHT[i])
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]                
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss
            
            if epoch < WARMUP_EPOCHS:
                lr = (epoch*self.spe + self.cs) / self.ws * LR_INIT
            else:
                lr = LR_INIT*0.9**((self.cs-self.spe)/200)
                if lr <= LR_END:
                    lr = LR_END
                
            self.optimizer.lr.assign(lr)
            

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            for i in range(len(gradients)):
                max_g = tf.reduce_max(tf.abs(gradients[i]))
                if max_g > 1.0:
                    print("NEED GRADIENT CLIP!!")
            
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            self.ave_losses[0] += giou_loss
            self.ave_losses[1] += conf_loss
            self.ave_losses[2] += prob_loss
            self.ave_losses[3] += total_loss

            self.lh.update(self.cs, self.optimizer.lr.numpy(),
                            giou_loss, conf_loss, prob_loss)
                    
            if self.cs + 1 == (epoch + 1) * self.spe:                             
                self.ave_losses /= self.spe
                
                tf.print("=> EPOCHS %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                "prob_loss: %4.2f   total_loss: %4.2f" %(epoch+1, self.optimizer.lr.numpy(),
                                                        self.ave_losses[0], self.ave_losses[1],
                                                        self.ave_losses[2], self.ave_losses[3]))

                best, last, stop = self.ls(epoch, self.ave_losses[3])
                if best:
                    print("Save best model at ",epoch)
                    self.save_model(BEST_MODEL)
                if last or stop :
                    self.save_model(LAST_MODEL)
                if stop:
                    self.save_model(LAST_MODEL)
                    self.set_learning_stop()
                    return 0
                 
                self.ave_losses = np.zeros_like(self.ave_losses)
                self.lh.save()
                
            self.cs += 1

    def _postprocess_label(self, label_paths):
        '''
        Arguments
            label_paths: (N, )               
        
        Return
            labels: (M, 5), (class, x_min, y_min, x_max, y_max), 
                
        '''        
        ori_h = self.input_size[0]
        ori_w = self.input_size[1]

        paths = label_paths.numpy()
        labels = []   
        for p in paths:
            label = np.loadtxt(p.decode(encoding='utf-8'))
            if np.shape(label) == (5,):
                label = label.reshape(1,5)            
            labels.extend(label)
        
        labels = np.array(labels)
        
        xywh = labels[:,1:]
        xywh[:,[0,2]] *= ori_w
        xywh[:,[1,3]] *= ori_h
        class_id = labels[:,1]

        # (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
        coor = np.concatenate([xywh[:,:2]-xywh[:,2:]*0.5, xywh[:,:2]+xywh[:,2:]*0.5], axis=-1)

        return np.concatenate([class_id[:, np.newaxis], coor], axis=-1)



    def nms(self, preds, iou_thres=0.45, sigma=0.3, method="nms"):
        """
        Argument
            preds: (3, )
                - preds[i] = [N, W, H, 3, 5 + n_class], (cx, cy, w, h, p, class)

        Return
            best_bbxoes: [M, 6], (x_min, y_min, x_max, y_max, conf, class)
        """
        # bboxes: (xmin, ymin, xmax, ymax, score, class)        
        bboxes = self._postprocess_bbox(preds)

        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bbox_ious(best_bbox[np.newaxis, :4], cls_bboxes[:, :4], mode="iou")
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_thres
                    # print("iou: ", np.shape(iou))
                    # print("mask: ", np.shape(iou_mask))
                    weight[tuple([iou_mask])] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
 
        return np.array(best_bboxes)

    def box_iou(self, box1, box2):
        """        
        Arguments:
            box1: [N, 4], (x_min, y_min, x_max, y_max)
            box2: [M, 4] (x_min, y_min, x_max, y_max)
        Returns:
            iou:  [N, M], the NxM matrix containing the pairwise
                IoU values for every element in preds and labels
        """        
        xy_min1, xy_max1 = tf.split(box1, 2, axis=1)
        xy_min2, xy_max2 = tf.split(box2, 2, axis=1)

        # inter = tf.minimum(xy_max1, xy_max2) - tf.minimum(xy_min1, xy_min2)
        # print(tf.math.minimum(xy_max1, xy_max2))
        # print("xy_max1: ", xy_max1)
        # print("xy_max2: ", xy_max2)
        # print(box1)

    def get_correction(self, preds, labels, iouv):
        '''
            preds: [N, 6], (x_min, y_min, x_max, y_max, prob, class)
                   result from "nms"
            labels: [M, 5] (class, x_min, y_min, x_max, y_max)
            iouv: iou vector for mAP50-95
        '''
        correct = np.zeros((np.shape(preds)[0], np.shape(iouv)[0])).astype(bool)

        preds_xyxy = preds[:,:4]        
        labels_xyxy = labels[:, 1:]
        # iou = self.bbox_ious(preds_xywh[:,np.newaxis,np.newaxis,np.newaxis,:4],
        #                     labels_xywh[:,np.newaxis,np.newaxis,np.newaxis,1:], 
        #                      mode="iou")
        
        iou = self.box_iou(preds_xyxy, labels_xyxy)

        # correct_class = labels[:,:1] == preds[:,5]

        # print(np.shape(preds[:,5]))
        # print(np.shape(labels[:,:1]))
        # print(correct_class)
        # print(np.shape(correct_class))
        # print(np.shape(iou))

        # for i in range(len(iouv)):
            # x = tf.where((iou >= iouv[i]) & correct_class)
            # print(np.shape(iou>=iouv[i]))
        #     print(np.shape(x))
            # if x[0].shape[0]:
                # print(np.shape(np.shape))
        #     print(x[0].shape[0])  

    def save_model(self, path):
        self.model.save_weights(path)

    def set_learning_stop(self):
        self.learning_stop = True
        
if __name__=="__main__":    
    path = DATA_PATH

    dl = DataLoader(path)
    im, ld, pd, ns = dl.get_dataset()    
    
    batch_size = 8
    nb = math.ceil(ns/batch_size) # number of batch
    
    learning_info = {}
    learning_info["steps_per_epoch"] = nb
    learning_info['epochs'] = EPOCHS
    
    model = HSMNIST((WIDTH, HEIGHT, 3), NUM_CLASS).build(training=True)
    trainer = Trainer(model, learning_info)

    im = im.batch(batch_size)
    ld = ld.batch(batch_size)
    pd = pd.batch(batch_size)
    
    # TODO:
    # 1. how to define best.pt?
    # 2. set early stopping
    #    - nms ??????(??????)
    #    - p, r, mAP ??????
    # 3. divide train/validation sets   

    # EPOCHS = 1 
    for epoch in range(EPOCHS):
        pbar = enumerate(zip(im, ld, pd))
        pbar = tqdm(pbar, total = nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for _, (c, l, p) in pbar:
            trainer.train_step(c, l, epoch)
            
            # preds = model(c)
            # best_bbox = trainer.nms(preds)

            # gt = trainer._postprocess_label(p)
            
            # iouv = tf.linspace(0.5, 0.95, 10)

            # trainer.get_correction(best_bbox, gt, iouv)
            # break
        # break
        if trainer.learning_stop:            
            break
        
    # trainer.save_model()
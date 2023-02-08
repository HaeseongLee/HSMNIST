import numpy as np
import matplotlib.pyplot as plt

# Record parameters through learning process per one epoch

class LearningHistory():
    def __init__(self, size):
        self.lr = np.zeros((size,1))
        self.gloss = np.zeros((size,1)) # giou loss
        self.closs = np.zeros((size,1)) # confidenc, or objectness loss
        self.ploss = np.zeros((size,1)) # probability loss
        self.tloss = np.zeros((size,1)) # total loss

        # TODO: add p,r,mAP etc.

    def update(self, i, lr, g, c, p):
        '''
            e: current epoch
        '''
        self.lr[i] = lr
        self.gloss[i] = g
        self.closs[i] = c
        self.ploss[i] = p
        self.tloss[i] = g + c + p

    def save(self, path="./learning_history.npy"):
        result = np.hstack((self.lr, self.gloss, self.closs, self.ploss, self.tloss))
        np.save(path, result)
        # print("Save learning result...")

        
if __name__ == "__main__":

    # lh = np.load("20221222_sgd_debug/learning_history.npy")
    lh = np.load("learning_history.npy")
    
    # lh2 = np.load("20230111_ciou_wo_mask_scaled/learning_history.npy")

    lr = lh[:,0]
    gloss = lh[:,1]
    closs = lh[:,2]
    ploss = lh[:,3]
    tloss = lh[:,4]

    # gloss2 = lh2[:,1]

    # plt.figure(figsize=(8,3))
    # plt.plot(gloss)
    # plt.plot(gloss2)
    # plt.show()

    print(min(tloss))
    plt.figure(figsize=(8,3))
    plt.subplot(1,4,1)    
    plt.plot(gloss)
    plt.title("giou")

    plt.subplot(1,4,2)    
    plt.plot(closs)
    plt.title("objectness")

    # plt.subplot(1,4,3)    
    # plt.plot(ploss)
    # plt.title("probability")


    plt.subplot(1,4,3)    
    plt.plot(lr[0:12])
    plt.title("learning rate")


    plt.subplot(1,4,4)    
    plt.plot(tloss)
    plt.title("total")

    plt.tight_layout()
    plt.show()

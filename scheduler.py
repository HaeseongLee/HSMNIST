import numpy as np

class EarlyStopping():
    def __init__(self, patience = 10):
        self.best_loss = 1000.0 # total loss, set initial value as large enough
        self.best_epoch = 0
        self.patience = patience # wait for stpes as "patience" to check improveness from the learning
    
    def call(self, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch

        delta = epoch - self.best_epoch

        stop = delta >= self.patience
        if stop:
            print("EARLY STOPPING!!")

        return stop

class LearningScheduler(EarlyStopping):
    def __init__(self, patience=10, save_setp=10):
        super().__init__(patience)
        self.save_step = save_setp # save the result per 10 epochs
        
    def __call__(self, epoch, loss):
        last = best = False

        if (epoch + 1) % self.save_step == 0:
            last = True

        if loss < self.best_loss:
            best = True
        
        stop = self.call(epoch, loss)

        return best, last, stop


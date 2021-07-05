import torch.nn as nn
import torch
from utils import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################
####### CLASSES

class Metrics(nn.Module):
    
    def __init__(self, metrics_names=[], num_eval=1):
        """
        Class to keep track of all the metrics in a model.
        Arguments:

        -metrics_names: iterable containing the names of the metrics to store (the epoch, times, loss_train, loss_val are added by default)
        it can be: 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'MatthewCorr'
        note that those metrics will be computed on the evaluation only.

        -num_eval (int): number of evaluations per epochs.

        Returns: Class Metrics
        The different elements of the class are:
        -epoch (int): number of epochs the model has been trained.
        -times (list(int)): times in seconds taken to train each epoch (len(times)==epoch).
        -loss_train (list(float)): loss on the training set at each epoch.
        -loss_val (list(float)): loss on th validation set at each epoch.
        """
        super(Metrics, self).__init__()

        self.num_eval = num_eval
        self.epoch = 0
        self.times = []
        self.loss_train = []
        self.loss_val = []
        self.steps = []
        self.metrics = {}

        for metric in metrics_names:
            self.metrics[metric] = []
    
    def forward(self):
        """
        Prints the model main informations
        """

        print(f"Model trained for {self.epoch} epochs.")

        if self.epoch > 0:
            print(f"Total train time: {seconds_to_string(sum(self.times))}.\n")
            print(f"Current metrics:")
            print("    -Train Loss: {:.4f}, Validation Loss: {:.4f}".format(self.loss_train[-1], self.loss_val[-1]))
            for metric in self.metrics.keys():
                print("    -{}: {:.4f}".format(metric, self.metrics[metric][-1]))
                
    def print(self):
        """
        Plots the metrics and losses (should be named plot(self))
        """
        
        if self.epoch == 0:
            print("Model not trained")
        else:
          
            x = self.steps

            nb_plots = len(self.metrics.keys())+1
            cols = 3
            rest = nb_plots % cols
            rows = nb_plots // cols if rest == 0 else nb_plots // cols + 1
            rows = max(2, rows)

            fig, axs = plt.subplots(rows, cols, figsize = (cols*5, rows*5))
            i, j = 0, 0

            axs[i,j].plot(x, self.loss_train, label='Train')
            axs[i,j].plot(x, self.loss_val, label = 'Validation')
            axs[i,j].set_title('Losses')
            axs[i,j].legend()

            j += 1

            for key in self.metrics.keys():
                axs[i,j].plot(x, self.metrics[key])
                axs[i,j].set_title(key)
                j += 1
                if j > cols-1:
                    i += 1
                    j = 0
            plt.show()

    def get_best(self, display = True):
        """
        Get the best value for each metric along with the epoch it has been attained
        Return:
            best_metrics (dict): dictionnary with [metric]: best_value and [metric + '_epoch']: epoch
        """

        best_metrics = self.metrics.copy()
        for key in self.metrics.keys():
            best_metrics[key] = max(self.metrics[key])
            best_metrics[key+"_epoch"] = torch.argmax(torch.tensor(self.metrics[key])).item()
            if display:
                print("Best {}: {:.4f} achieved at epoch {} in {}".format(key, best_metrics[key], best_metrics[key+"_epoch"]+1,
                                                                        seconds_to_string(sum(self.times[:best_metrics[key+"_epoch"]]))))
        return best_metrics
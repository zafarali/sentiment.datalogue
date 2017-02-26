import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys, os

def training_plot(history, outfile, metric='categorical_accuracy'):
    """
    Plot training accuracy for each epoch
    """
    ## Set output file for plot
    basepath = os.path.split(os.path.expanduser(outfile))[0]
    plotfile = basepath + '_train_plot.png'

    ## Plot accuracy
    plt.plot(history.history['val_'+metric], label='test')
    plt.plot(history.history[metric], label='train')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plotfile)
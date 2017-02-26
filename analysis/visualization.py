import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sns.set_style('white')

def training_plot(history, outfile, metric='categorical_accuracy', title=''):
    """
    Plot training accuracy for each epoch
    """
    ## Set output file for plot
    basepath = os.path.split(os.path.expanduser(outfile))[0]
    plotfile = basepath + '_train_plot.png'

    ## Plot accuracy
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(history.history['val_'+metric], label='test')
    ax.plot(history.history[metric], label='train')
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric)
    ax.legend()
    f.savefig(plotfile)
    return f, ax


def plot_single_auc(fpr, tpr, auc_, ax=None, c='b', label=''):
    """
    Plots the receiver operating characteristic curve for a single 
    sequence of false positive rates, true postive rates and auc
    """
    ax_ = ax
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    ax.plot(fpr, tpr, lw=2, color=c,\
        label=label + ' AUC:' + str(auc_) )

    if ax_ is None:
        return f, ax
    else:
        return ax


def plot_auc(fprs, tprs, aucs, title='Receiver Operating Characteristc', labels=None):
    assert len(fprs) == len(tprs), 'must have equal number of FPRs and TPRS'
    assert len(tprs) == len(aucs), 'must have equal number of tprs and aucs'

    COLORS = sns.color_palette(n_colors=len(aucs))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    labels = [''] * len(aucs) if not labels else labels
    assert len(labels) == len(aucs), 'must have equal number of labels as aucs'

    # should probably be more descirptive with variable names...
    for f, t, a, c, l in zip(fprs, tprs, aucs, COLORS, labels):
        plot_single_auc(f, t, a, ax=ax, c=c, label= l)

    ax.plot([0, 1], [0, 1], lw=2, linestyle='--', color='k', label='Random')
    ax.set_xlabel('false positive rates')
    ax.set_ylabel('true positive rates')
    ax.legend()
    ax.set_title(title)

    return fig, ax



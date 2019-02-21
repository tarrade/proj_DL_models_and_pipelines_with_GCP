
import matplotlib.pyplot as plt
import gzip
import _pickle as cPickle
import sys

def plot_mnist_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 25

    # create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(5, 5, figsize=(18, 15))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        # plot image
        # ax.imshow(images[i], cmap='binary')
        ax.imshow(images[i], cmap=plt.cm.gray)

        # show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # ensure the plot is shown correctly with multiple plots in a single Notebook cell.
    plt.show()


def load_data(path):
    # get mnist data, split between train and test sets
    f = gzip.open(path, 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
    f.close()
    return data
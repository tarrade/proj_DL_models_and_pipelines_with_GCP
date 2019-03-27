import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from tensorboard.backend.event_processing import event_accumulator

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.style.use('seaborn-ticks')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(15 ,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=20)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label');
    plt.xlabel('Predicted label');


def roc_curves(y_test, y_score, dict_label):

    plt.style.use('seaborn-ticks')

    # plot configuration
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 15})
    lw = 2

    n_classes = len(dict_label)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # compute macro-average ROC curve and ROC area

    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Ffnally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for multi-class')
    plt.legend(loc="best")
    plt.show()

    # plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['pink', 'purple', 'deeppink', 'lavenderblush', 'darkorchid', 'orchid', 'hotpink', 'darkslateblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve for {0} (area = {1:0.2f})'
                       ''.format(dict_label[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for multi-class')
    plt.legend(loc="best", prop={'size': 13})
    plt.show()

    # zoom in view of the upper left corner.
    plt.figure(figsize=(10, 8))
    plt.xlim(0, 0.4)
    plt.ylim(0.7, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['pink', 'purple', 'deeppink', 'lavenderblush', 'darkorchid', 'orchid', 'hotpink', 'darkslateblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve for {0} (area = {1:0.2f})'
                       ''.format(dict_label[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for multi-class')
    plt.legend(loc="best")
    plt.show()


def load_data_tensorboard(path):
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    data = {}

    for tag in sorted(event_acc.Tags()["scalars"]):
        x, y = [], []
        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.step)
            y.append(scalar_event.value)
        data[tag] = (np.asarray(x), np.asarray(y))
    return data

def plot_acc_loss(steps_loss_train, loss_train,
                  steps_acc_train=None, accuracy_train=None,
                  steps_loss_eval=None, loss_eval=None,
                  steps_acc_eval=None, accuracy_eval=None):

    # plot the training loss and accuracy
    fig = plt.figure(figsize=(9, 3), dpi=100)
    plt.subplots_adjust(wspace=0.6)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    # accuracy
    if accuracy_train is not None:
        ax1.plot(steps_acc_train, accuracy_train, 'b', label='training accuracy')
    if accuracy_eval is not None:
        ax1.plot(steps_acc_eval, accuracy_eval, 'r', label='validation accuracy');
    ax1.set_title('Accuracy')
    ax1.set_xlabel("Number of epoch ")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="best")
    # loss
    if loss_train is not None:
        ax2.plot(steps_loss_train, loss_train, label="training loss")
    if loss_eval is not None:
        ax2.plot(steps_loss_eval, loss_eval, label="validation loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Number of epoch ")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="best");

    print('Loss:')
    if loss_train is not None:
        print('  - loss [training dataset]: {0:.3f}'.format(loss_train[-1]))
    if loss_eval is not None:
        print('  - loss [validation dataset: {0:.3f}'.format(loss_eval[-1]))
    print('')
    print('Accuracy:')
    if accuracy_train is not None:
        print('  - accuracy [training dataset]: {:.2f}%'.format(100 * accuracy_train[-1]))
    if accuracy_eval is not None:
        print('  - accuracy [validation dataset: {:.2f}%'.format(100 * accuracy_eval[-1]))
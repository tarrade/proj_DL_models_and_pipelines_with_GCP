from src.pkg_mnist_fnn.utils import load_data
from src.pkg_mnist_fnn.model import parse_images
from src.pkg_mnist_fnn.model import parse_labels

def parse_images(x):
    return x.reshape(len(x), -1)

import numpy as np
import pandas as pd
(x_train, y_train), (x_test, y_test) = load_data("data/mnist/raw/")

X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

X = parse_images(X)
#Y = parse_labels(Y)

labels_pixel = ['feat_{}'.format(i) for i in range(1, X.shape[-1]+1)]

X = pd.DataFrame(X, columns=labels_pixel)
X["label"] = Y

X.to_csv("data/mnist/raw/mnist_all.csv", index=True, index_label='ID')

tab_images = X.label.value_counts()
with open("data/mnist/mnist_number_tabulate.tex", "w") as f:
    tab_images.to_latex(buf=f)
with open("data/mnist/mnist_number_tabulate", "w") as f:
    tab_images.to_string(buf=f)

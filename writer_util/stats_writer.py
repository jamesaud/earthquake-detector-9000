
import scikitplot as skplt
import os; pj = os.path.join
from functools import wraps
import matplotlib.pyplot as plt
import pathlib


def save_figure(fn, path):
    """ Decorator that Saves the figure output by a function in the skplt library """
    @wraps(fn)
    def save_fig(*args, **kwargs):
        figure = fn(*args, **kwargs)
        plt.savefig(path)
        return figure

    return save_fig


class StatsWriter:

    def __init__(self, path):
        self.path = path

        # Create directory if it doesn't exist
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Methods, with a specific folder for writing
        self.write_roc = save_figure(skplt.metrics.plot_roc, pj(path, 'roc.png'))
        self.write_confusion_matrix = save_figure(skplt.metrics.plot_confusion_matrix, pj(path, 'confusion_matrix.png'))
        self.write_pr = save_figure(skplt.metrics.plot_precision_recall, pj(path, 'pr_curve.png'))


    def write_stats(self, true_labels, output_labels, predicted_labels):
        print(true_labels)
        print(output_labels)
        self.write_roc(true_labels, output_labels)
        self.write_confusion_matrix(true_labels, predicted_labels)
        self.write_pr(true_labels, output_labels)
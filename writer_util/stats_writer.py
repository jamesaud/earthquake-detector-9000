
import scikitplot as skplt
import os; pj = os.path.join
from functools import wraps
import matplotlib.pyplot as plt
import pathlib


def save_figure(fn, path):
    """ Decorator that Saves the figure output by a function in the skplt library """
    @wraps(fn)
    def save_fig(*args, **kwargs):
        ax = fn(*args, **kwargs)
        plt.savefig(path)
        return ax

    return save_fig


class StatsWriter:
    # TODO: Make the class_names an attribute and generalize to the other legend writing functions

    def __init__(self, path):
        self.path = path

        # Create directory if it doesn't exist
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Methods, with a specific folder for writing


    def write_roc(self, true_labels, output_probabilities):
        ax = skplt.metrics.plot_roc(true_labels, output_probabilities, plot_micro=False, plot_macro=False)
        legend = plt.legend()

        for label in legend.get_texts():
            text = label.get_text()
            text = text.replace("class 0", "class Noise")
            text = text.replace("class 1", "class Event")
            label.set_text(text)

        plt.savefig(pj(self.path, 'roc.png'))

    def write_confusion_matrix(self, true_labels, predicted_labels):
        skplt.metrics.plot_confusion_matrix(true_labels, predicted_labels)
        plt.savefig(pj(self.path, 'confusion_matrix.png'))

    def write_pr(self, true_labels, output_probabilities):
        ax = skplt.metrics.plot_precision_recall(true_labels, output_probabilities, plot_micro=False)

        legend = plt.legend()
        for label in legend.get_texts():
            text = label.get_text()
            text = text.replace("class 0", "class Noise")
            text = text.replace("class 1", "class Event")
            label.set_text(text)

        plt.savefig(pj(self.path, 'pr_curve.png'))


    def write_stats(self, true_labels, output_probabilities, predicted_labels, class_names=None):
        """ 
        All numpy arrays work well as arguments 
        :class_names: dictionary mapping the class number to a human readable name for the axis titles
        """
        if class_names:
            named_true_labels = [class_names[val] for val in true_labels]
            named_predicted_labels = [class_names[val] for val in predicted_labels]
            
        self.write_roc(true_labels, output_probabilities)
        self.write_confusion_matrix(named_true_labels, named_predicted_labels)
        self.write_pr(true_labels, output_probabilities)

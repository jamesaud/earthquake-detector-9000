import torch
from torch.autograd import Variable

from utils import dotdict
import os
import sys
import copy

class Evaluator:

    def __init__(self, true_labels, output_labels, num_classes):
        self.class_info = dotdict()  # Javasript style item access

        # Should be updated as arrays
        self.true_labels = true_labels         # True labels
        self.output_labels = output_labels     # Raw output classes of neural net
        self.num_classes = num_classes

        _, self.predicted_labels = torch.max(output_labels.data, 1)   # Label with the highest score

        self.compute_accuracy(self.predicted_labels, self.true_labels, self.num_classes)

    def compute_accuracy(self, predicted_labels, true_labels, num_classes):
        """
        :param predicted: 1d Tensor: An array of predictions (probabilities for each label)
        :param predicted: 1d Tensor: the predicted lab
        :param class_labels: List: the class labels
        :return:
        """
        class_correct = [0 for _ in range(num_classes)]
        class_total = [0 for _ in range(num_classes)]

        guesses = (predicted_labels == true_labels).squeeze()

        for guess, label in zip(guesses, true_labels):
            guess, label = guess.item(), label.item()
            class_correct[label] += guess
            class_total[label] += 1

        for i, (correct, total) in enumerate(zip(class_correct, class_total)):
            self.update_accuracy(class_name=i, amount_correct=correct, amount_total=total)


    def update_accuracy(self, class_name: str, amount_correct: int, amount_total: int):
        self.class_info[class_name] = {
            'amount_correct': amount_correct,
            'amount_total': amount_total
        }

    def class_details(self, class_name):
        return self.class_info[class_name]

    def percent_correct(self, class_name):
        Class = self.class_info[class_name]
        try:
            return Class.amount_correct / Class.amount_total
        except ZeroDivisionError:
            return 0

    def total_percent_correct(self):
        amount_correct = 0
        amount_total = 0

        for class_name, info in self.class_info.items():
            amount_correct += info.amount_correct
            amount_total += info.amount_total

        try:
            return amount_correct / amount_total
        except ZeroDivisionError:
            return 0
    
    def normalized_percent_correct(self, weigh_events=1):
        return (self.percent_correct(0) + self.percent_correct(1) * weigh_events) / (1 + weigh_events)

    def __str__(self):
        return 'Evaluator Object: ' + str(self.class_info)


class NetEval:

    def __init__(self, net):
        self.net = net

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs

    def predict(self, inputs):
        outputs = self.forward(inputs)
        predicted_labels = self.predicted_classes(outputs)
        return predicted_labels

    def predicted_classes(self, outputs):
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def correct_predictions(self, true_labels, predicted_labels):
        guesses = (predicted_labels == true_labels).squeeze()
        return guesses

    def to_cuda(self, inputs):
        return [Variable(input).cuda() for input in inputs]


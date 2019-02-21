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

        print(output_labels.data)
        _, predicted_labels = torch.max(output_labels.data, 1)
        predicted_labels = predicted_labels    # Labels projected to the nearest class
        self.predicted_labels = predicted_labels

        self.compute_accuracy(predicted_labels, true_labels, num_classes)

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
    
    def normalized_percent_correct(self):
        return (self.percent_correct(0)*1.1 + self.percent_correct(1)) / 2.1

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


def evaluate(net, data_loader, num_classes, BATCH_SIZE):
    """
    Goes through entire loader one time
    :param net: neural net
    :param copy_net: boolean
    :param data_loader: DataLoader
    :return: Data structure of class Evaluator containing the amount correct for each class
    """
    Net = net
    net_eval = NetEval(Net)
    eval = Evaluator()

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    i = 0 
    size = BATCH_SIZE * len(data_loader)

    for (inputs, labels) in data_loader:
        inputs, labels = net_eval.to_cuda(inputs), labels.cuda()
        guesses = net_eval.predict(inputs)

        for guess, label in zip(guesses, labels):
            class_correct[label] += guess
            class_total[label] += 1

        i += BATCH_SIZE
        sys.stdout.write('\r' + str(i) + '/' + str(size))
        sys.stdout.flush()

    # Update the information in the Evaluator
    for i, (correct, total) in enumerate(zip(class_correct, class_total)):
        eval.update_accuracy(class_name=i, amount_correct=correct, amount_total=total)

    return eval



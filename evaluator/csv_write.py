#### WRITING CSV RESULTS FOR TESTING A NET ####
import csv
from .evaluator import NetEval
from loaders.named_loader import AbstractSpectrogramNamedDataset
import sys


def print_progress(i):
    sys.stdout.write('\r' + str(i))
    sys.stdout.flush()

def write_new_csv_headers(csv_file, headers):
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)   # Write Headers


def write_csv_predictions(guesses, labels, csv_file):
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)

        for guess, label in zip(guesses, labels):
            guess = guess[0]
            writer.writerow([guess, label, guess==label])


def write_predictions_to_csv(net, data_loader, csv_path):
    """
    Goes through entire loader one time
    :param net: neural net
    :param copy_net: boolean
    :param data_loader: DataLoader
    :return: Data structure of class Evaluator containing the amount correct for each class
    """
    Net = net
    net_eval = NetEval(Net)
    write_new_csv_headers(csv_path, ['Guesses', 'True Labels', 'Correct'])

    i = 0
    for (inputs, labels) in data_loader:
        inputs, labels = net_eval.to_cuda(inputs), labels.cuda()
        guesses = net_eval.predict(inputs)

        write_csv_predictions(guesses, labels, csv_path)
        i += len(guesses)
        print_progress(i)


def write_named_csv_prediction(names, guesses, true_labels, csv_file):
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)

        for guess, true_label, name in zip(guesses, true_labels, names):
            guess, true_label, name = guess.item(), true_label.item(), name
            writer.writerow([name, guess, true_label])


def write_named_predictions_to_csv(net, named_dataloader: AbstractSpectrogramNamedDataset, csv_path):
    """
    Writes the Name, Guess, and True Label to a CSV file
    """
    Net = net
    net_eval = NetEval(Net)
    write_new_csv_headers(csv_path, ['Name', 'Guess', 'True Label'])

    i = 0
    for (inputs, true_labels, names) in named_dataloader:
        inputs = net_eval.to_cuda(inputs)
        guesses = net_eval.predict(inputs)

        write_named_csv_prediction(names, guesses, true_labels, csv_path)

        i += len(guesses)
        print_progress(i)

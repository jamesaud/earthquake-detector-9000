#### WRITING CSV RESULTS FOR TESTING A NET ####
import csv
from .evaluator import NetEval
import sys

def write_new_csv_headers(csv_file):
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Guesses', 'True Labels', 'Correct'])   # Write Headers
        

def write_csv_predictions(guesses, true_labels, csv_file):
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)

        for guess, label in zip(guesses, true_labels):
            guess = guess[0]
            writer.writerow([guess, label, guess==label])


def write_predictions_to_csv(net, data_loader, num_classes, csv_path):
    """
    Goes through entire loader one time
    :param net: neural net
    :param copy_net: boolean
    :param data_loader: DataLoader
    :return: Data structure of class Evaluator containing the amount correct for each class
    """
    Net = net
    net_eval = NetEval(Net)
    write_new_csv_headers(csv_path)

    i = 0 
    for (inputs, labels) in data_loader:
        inputs, labels = net_eval.to_cuda(inputs), labels.cuda()
        guesses = net_eval.predict(inputs, labels)

        write_csv_predictions(guesses, labels, csv_path)
        i += len(guesses)
        sys.stdout.write('\r' + str(i))
        sys.stdout.flush()
#### WRITING CSV RESULTS FOR TESTING A NET ####
import csv
from .evaluator import NetEval
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



def write_unknown_csv_prediction(names, guesses, csv_file):
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)

        for guess, name in zip(guesses, names):
            guess, name = guess.item(), name.item()
            writer.writerow([name, guess])

def write_unknown_predictions_to_csv(net, unknown_data_loader, csv_path):
    Net = net
    net_eval = NetEval(Net)
    write_new_csv_headers(csv_path, ['Guess', 'Guess'])

    i = 0 
    for (inputs, names) in unknown_data_loader:
        inputs = net_eval.to_cuda(inputs)
        guesses = net_eval.predict(inputs)

        write_unknown_csv_prediction(names, guesses, csv_path)

        i += len(guesses)
        print_progress(i)

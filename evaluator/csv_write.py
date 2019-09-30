#### WRITING CSV RESULTS FOR TESTING A NET ####
import csv
from .evaluator import NetEval
from loaders.named_loader import AbstractSpectrogramNamedDataset
import sys
import os 

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
            guess = guess.data[0]
            #guess, true_label, name = guess.item(), true_label.item(), name   # Pytorch 0.4+
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

def write_evaluator(evaluator, csv_path, extra_header_data=[], extra_row_data=[]):
    headers = [] + extra_header_data
    row = [] + extra_row_data

    
    for class_name, data in evaluator.class_info.items():
        for description, value in data.items():
            headers.append(description + f" (class {class_name})")
            row.append(value)

    for class_name, _ in evaluator.class_info.items():
        percent_correct = evaluator.percent_correct(class_name)
        headers.append(f"Percent Correct (class {class_name})")
        row.append(percent_correct)

    headers.append("Total Percent Correct")
    row.append(evaluator.normalized_percent_correct())

    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        if not exists:
            writer.writerow(headers)
        writer.writerow(row)
            
            
            
            


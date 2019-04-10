from validator import data_validator as dv
from importlib import reload
import config
import main
import os
import pathlib
import csv
import pandas as pd

folder = 'all-spectrograms-symlinks/99.5'
cwd = os.getcwd()
spectrogram_path = os.path.join(cwd, 'data', folder)

# Use "config_crossvalidation"  if doing cross validation
file = 'validator/config_crossvalidation.json'
validator_path = 'validator'
csv_file = 'weighted_results_cross_validation.csv'
csv_path = os.path.join(validator_path, csv_file)
epochs = 10


def get_name(path):
    path = pathlib.Path(path)  # ensures standard format of paths
    path = os.path.basename(path)  # get folder name
    return os.path.join(folder, path)

def write_to_csv(name, noise_correct, noise_total, local_correct, local_total, total_percent_correct, epochs):
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([name, 
                        noise_correct, 
                        noise_total, 
                        local_correct, 
                        local_total, 
                        total_percent_correct,
                        epochs])
                        


def initialize_csv(csv_path):
    if not os.path.exists(csv_path):
        write_to_csv('Name', 
                    'Amount Correct Noise', 
                    'Amount Total Noise', 
                    'Amount Correct Local', 
                    'Amount Total Local', 
                    'Total Percent Correct',
                    'Epochs')

def reset():
    """
    Jank code - Reloads the appropriate modules so that the environment variable is reread in the config file, and the neural net is completely reset
    """
    reload(config)
    reload(main)
    reload(dv)


def run_environ(skip_existing=True):
    os.environ['CONFIGURATION'] = file
    dirs = dv.get_paths(spectrogram_path)
    dirs = [get_name(dir) for dir in dirs]

    # Removes paths already ran (checks csv)
    if skip_existing:
        df = pd.read_csv(csv_path)
        names = list(map(get_name, df['Name'].tolist()))
        print(f"Skipping {len(names)} directories already in CSV")
        dirs = list(set(dirs).difference(set(names)))

    configuration = dv.read_config(file)
    initialize_csv(csv_path)

    for i, dir in enumerate(dirs):
        print("\n", "-" * 20)
        print(f"Progress: {i}/{len(dirs)}")
        print("current dir " + dir)
        print("-" * 20, "\n")

        dv.update_config(configuration, dir)
        dv.write_config(file, configuration)
        
        try:
            # Reload environment variables and main file with new configuration
            reset()
            main.print_config()

            evaluator = dv.test_best_dataset(epochs)
            
            print('\n', evaluator, '\n')
            write_to_csv(dir, 
                         evaluator.class_details(0).amount_correct,
                         evaluator.class_details(0).amount_total,
                         evaluator.class_details(1).amount_correct,
                         evaluator.class_details(1).amount_total,
                         str(evaluator.total_percent_correct()),
                         epochs,
                        )
       
        except Exception as e:
            print("Failed to run neural net: ", e)

def run_cross_validation():
    """Leave one out cross validation"""
    import copy

    os.environ['CONFIGURATION'] = file
    dirs = list(map(os.path.basename, 
                    map(get_name, dv.get_paths(spectrogram_path))))

    configuration = dv.read_config(file)
    initialize_csv(csv_path)

    for i, dir in enumerate(dirs):
        print("\n", "-" * 20)
        print(f"Progress: {i}/{len(dirs)}")
        print("-" * 20, "\n")
        
        ignore_test = copy.copy(dirs)
        ignore_test.remove(dir)

        configuration['train']['ignore'] = [dir]
        configuration['test']['ignore'] = ignore_test

        dv.write_config(file, configuration)
        
        try:
            # Reload environment variables and main file with new configuration
            reset()
            print("Evaluating Net on " + dir)
            evaluator = dv.test_best_dataset(epochs=1, evaluate_every=50000)
            
            print('\n', evaluator, '\n')
            write_to_csv(dir, 
                    evaluator.class_details(0).amount_correct, 
                    evaluator.class_details(0).amount_total, 
                    evaluator.class_details(1).amount_correct, 
                    evaluator.class_details(1).amount_total, 
                    str(evaluator.total_percent_correct()),
                    evaluator.iteration,
                    )
       
        except Exception as e:
            print("Failed to run neural net: ", e)
            raise

if __name__ == '__main__':
    import glob
    run_cross_validation()
    # run_environ(skip_existing=True)           # Make sure 'environment' is set as configuration in main.py

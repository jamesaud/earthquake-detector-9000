from validator import data_validator as dv
from importlib import reload
import config
import main
import os
import pathlib
import csv


folder = 'everywhere-untested'
cwd = os.getcwd()
spectrogram_path = os.path.join(cwd, 'data', folder)
file = 'validator/config.json'
validator_path = 'validator'
csv_file = 'results_everywhere.csv'
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


def run_environ():
    os.environ['CONFIGURATION'] = file
    dirs = dv.get_paths(spectrogram_path)
    configuration = dv.read_config(file)
    initialize_csv(csv_path)

    for dir in dirs:
        dir = get_name(dir)
        dv.update_config(configuration, dir)
        dv.write_config(file, configuration)
        
        # Reload environment variables and main file with new configuration
        reset()
        try:
            print("Training Net on " + dir)
            evaluator = dv.test_dataset(epochs)
            
        except Exception as e:
            print("Failed to run neural net: ", e)

        else:
            print('\n', evaluator, '\n')
            write_to_csv(dir, 
                    evaluator.class_details(0).amount_correct, 
                    evaluator.class_details(0).amount_total, 
                    evaluator.class_details(1).amount_correct, 
                    evaluator.class_details(1).amount_total, 
                    str(evaluator.total_percent_correct()),
                    epochs,
                    )
       


if __name__ == '__main__':
    run_environ()
from validator import data_validator as dv
from importlib import reload
import config
import main
import os
import pathlib
import csv
import pandas as pd
from pytorch_utils.utils import train_best_model, evaluate

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


def new_cv(skip_existing=True):
    import copy

    configuration = dv.read_config(file)
    configuration['train']['path'] = folder
    configuration['test']['path'] = folder

    dirs = list(map(os.path.basename, 
                    map(get_name, dv.get_paths(folder))))
    
    initialize_csv(csv_path)
    if skip_existing:
        df = pd.read_csv(csv_path)
        names = list(map(get_name, df['Name'].tolist()))
        print(f"Skipping {len(names)} directories already in CSV")
        dirs = list(set(dirs).difference(set(names)))
    

    for i, dir in enumerate(dirs):
        print("\n", "-" * 20)
        print(f"Progress: {i}/{len(dirs)}")
        print("-" * 20, "\n")

        net, optimizer, criterion = main.create_model()
        
        ignore_test = copy.copy(dirs)
        ignore_test.remove(dir)
        configuration['train']['ignore'] = [dir]
        configuration['test']['ignore'] = ignore_test

        dataset_train = main.create_dataset(configuration, model.transformations['train'], train=True)
        dataset_test = main.create_dataset(configuration, model.transformations['test'], train=False)

        # Create a final test loader where it has unseen data, by taking 10% of the training data
        dataset_final = copy.deepcopy(dataset_train)
        ntest_samples = len(dataset_train) * .1
        del dataset_final.file_paths[ntest_samples:]
        del dataset_train.file_paths[:ntest_samples]

        assert verify_dataset_integrity(dataset_train, dataset_test)
        assert verify_dataset_integrity(dataset_train, dataset_final)

        weigh_classes = dict(enumerate(configuration['weigh_classes']))
        train_loader = create_loader(dataset_train, train=True, weigh_classes=weigh_classes)
        test_loader = create_loader(dataset_test, train=False)
        final_test_loader =  create_loader(dataset_final, train=False)

        try:
            # Reload environment variables and main file with new configuration
            print("Evaluating Net on " + dir)
            evaluator, best_epoch = train_best_model(epochs=1,
                                                     train_loader, test_loader,net, optimizer, criterion, net,
                                                     writer=writer,
                                                     write=False,
                                                     yield_every=50_000)
            
            evaluator = evaluate(net, loader, copy_net=copy_net)
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
    run_cv()
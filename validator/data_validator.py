import glob
import json
from main import *  # Yeah... need to improve how this is done

from pytorch_utils.utils import evaluate, train_epoch
from evaluator.evaluator import Evaluator

def test_dataset(epochs):

    for epoch in range(epochs):
        train_epoch(epoch)

    evaluator: Evaluator = evaluate(net, test_loader)
    return evaluator


def test_best_dataset(epochs, evaluate_every):
    best = None
    write_initial(writer, net, settings, resize, crop, dataset_train)

    for epoch in range(epochs):
        # Will return evaluator every epoch

        trainer = train_epoch(epoch, train_loader, test_loader, optimizer, criterion, net, writer,
                              write=True, yield_every=evaluate_every)

        for evaluator in trainer:
            if best is None:
                best = evaluator

            if evaluator.normalized_percent_correct(weigh_events=1.1) >= best.normalized_percent_correct(weigh_events=1.1):
                best = evaluator
                best.iteration = iterations

    return best

def get_paths(path):
    dirs = glob.glob(os.path.join(path, '*'))
    return dirs


def write_config(file, dic):
    with open(file, 'w') as fle:
        fle.write(json.dumps(dic))


def read_config(file):
    with open(file, 'r') as fle:
        return json.loads(fle.read())


def update_config(dic, path):
    dic['train']['path'] = path
    dic['test']['path'] = path
    return dic



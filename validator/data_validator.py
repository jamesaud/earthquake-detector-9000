import glob
import json
from main import *
from evaluator.evaluator import NetEval
import csv
import copy

def test_dataset(epochs, write=False):

    for epoch in range(epochs):
        train(epoch, write=write)

    evaluator: Evaluator = evaluate(net, test_loader)
    return evaluator


def test_best_dataset(epochs, write=False):
    best = None

    for epoch in range(epochs):
        for evaluator in train(epoch, write=write, yield_evaluator=True):
            if best is None:
                best = evaluator
            
            if evaluator.normalized_percent_correct() >= best.normalized_percent_correct():
                best = evaluator
                best.iteration = iterations 

        if iterations > 200000:
            return best
    return best

def get_paths(path):
    dirs = glob.glob(os.path.join(path, '*/'))
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



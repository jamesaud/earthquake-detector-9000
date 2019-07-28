import copy
import os
import sys
from functools import wraps

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluator.evaluator import Evaluator
from writer_util import MySummaryWriter as SummaryWriter
from itertools import islice
import math
from .data_utils import subsample_dataset, replace_loader_dataset

import sys
sys.path.append('..')
from evaluator import csv_write  


def evaluate(net: nn.Module,
             data_loader: DataLoader,
             copy_net=False):
    """
    :param net: neural net
    :param copy_net: boolean
    :param data_loader: DataLoader
    :return: Data structure of class Evaluator containing the amount correct for each class
    """
    if copy_net:
        Net = copy.deepcopy(net)
        Net.eval()
    else:
        Net = net

    i = 0
    all_output_labels = torch.FloatTensor()
    all_true_labels = torch.LongTensor()

    for (inputs, labels) in data_loader:
        inputs, labels = [Variable(input).cuda() for input in inputs], labels
        output_labels = Net(inputs).cpu().detach().data

        all_output_labels = torch.cat((all_output_labels, output_labels))
        all_true_labels = torch.cat((all_true_labels, labels))

        i += data_loader.batch_size
        sys.stdout.write('\r' + str(i) + '/' + str(data_loader.batch_size * len(data_loader)))
        sys.stdout.flush()


    # Update the information in the Evaluator
    eval = Evaluator(all_true_labels, all_output_labels, 2)
    return eval


def write_images(writer, dataset_train):
    """
    :return:
    """
    # Write Images
    noise_index = next(dataset_train.get_next_index_with_label(0))
    local_index = next(dataset_train.get_next_index_with_label(1))

    noise_raw = dataset_train.preview_raw(noise_index, show=False)
    img1 = writer.figure_to_image(noise_raw)

    noise_transformed = dataset_train.preview(noise_index, show=False)
    img2 = writer.figure_to_image(noise_transformed)

    local_raw = dataset_train.preview_raw(local_index, show=False)
    img3 = writer.figure_to_image(local_raw)

    local_transformed = dataset_train.preview(local_index, show=False)
    img4 = writer.figure_to_image(local_transformed)

    noise_image = writer.combine_images_horizontal([img1, img2])
    local_image = writer.combine_images_horizontal([img3, img4])

    image = writer.combine_images_vertical([noise_image, local_image])
    writer.add_pil_image('Transformations', image)


def save_model(path, net):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)

def save_checkpoint(path, name, model, optimizer, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
                'name': name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)

# Use existing model
def load_checkpoint(checkpoint_path, model, optimizer):
        checkpoint = torch.load(checkpoint_path)            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

def load_model(net: nn.Module, path: str):
    return net.load_state_dict(torch.load(path))


def print_evaluation(evaluator, description):
    correct = 100 * evaluator.total_percent_correct()
    print('Accuracy of the network on %s images: %d %%' % (description, correct))

    for name, info in evaluator.class_info.items():
        print('Accuracy of the network on %s class  %s: [%2d / %2d] %2d %%' % (description, name,
                                                                               info.amount_correct,
                                                                               info.amount_total,
                                                                               evaluator.percent_correct(name) * 100))


def write_histogram(writer, net, iterations):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), iterations)


def write_info(writer: SummaryWriter, net: nn.Module, settings: dict, resize: tuple, crop: tuple):
    transforms = ['Resize: ' + str(resize), 'Crop: ' + str(crop)]

    step = 0
    for transformation in transforms + net._train + net._transformations:
        writer.add_text('Transformations_Train', str(transformation), global_step=step)
        step += 1

    step = 0
    for transformation in transforms + net._test + net._transformations:
        writer.add_text('Transformations_Test', str(transformation), global_step=step)
        step += 1

    writer.add_text('Configuration', str(settings))


def write_loss(writer, loss, iterations):
    writer.add_scalar('train_loss', loss, iterations)


def evaluation(net: nn.Module,
              loader: DataLoader,
              name: str,
              copy_net: bool = True):
    """
    :param net: Torch Model
    :param loader: Dataloader
    :return: Evaluator containing the stats for this evaluation
    """
    print(f"\nTesting for {name}...")
    evaluator = evaluate(net, loader, copy_net=copy_net)
    print()
    print_evaluation(evaluator, name)
    return evaluator


def write_evaluator(writer, name, evaluator, iterations):
    writer.add_scalars('amount_correct',
                       {f'{name}_amount_correct': evaluator.normalized_percent_correct()},
                       iterations)

    writer.add_scalars(f'{name}_class_correct',
                       {f'{name}_noise': evaluator.percent_correct(0),
                        f'{name}_local': evaluator.percent_correct(1)},
                       iterations)


def print_loss(batch_num, loss, epoch, train_loader):
    total_samples = len(train_loader) * train_loader.batch_size
    batch_samples = batch_num * train_loader.batch_size
    ratio = batch_samples / total_samples
    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch,
        int(batch_samples),
        int(total_samples),
        float(ratio) * 100,
        float(loss))

    sys.stdout.write('\r' + msg)
    sys.stdout.flush()



def train_batches(train_loader: DataLoader,
                net: nn.Module,
                optimizer,
                criterion):
    """
    yields the loss every batch
    :param train_loader: Dataloader
    :param net: Pytorch Model
    :param optimizer:
    :param criterion:
    :return: dictionary containing stats about the training
    """
    for i, (true_inputs, true_labels) in enumerate(train_loader):
        # wrap them in Variable
        inputs, labels = [Variable(input).cuda() for input in true_inputs], Variable(true_labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # backpropagate and update optimizer learning rate
        loss.backward()
        optimizer.step()

        yield {"loss": loss.data}


def train_epoch(epoch: int,
                train_loader: DataLoader,
                test_loader: DataLoader,
                optimizer,
                criterion,
                net: nn.Module,
                writer: SummaryWriter,
                write=True,
                checkpoint_path=None,
                yield_every=float('inf'),
                print_loss_every=1000,
                print_test_evaluation_every=float('inf'),
                print_train_evaluation_every=float('inf'),
                train_evaluation_loader: DataLoader = None,
                evaluate_at_end=True):
    """

    :param epoch: The epoch number
    :param train_loader: Pytorch train loader
    :param test_loader:  Pytorch test loader
    :param net: Pytorch Model
    :param write:  Whether to write stats or not
    :param writer: Summarywriter to write information
    :param write: Whether or not to write
    :param checkpoint_path: Where to write models
    :param yield_every: How many iterations to yield  the test evaluator
    :param print_loss_every: When to print and write loss
    :param print_test_evaluation_every:  when to print and write test evaluation
    :param print_train_evaluation_every:  when to print and write train_evaluation
    :param train_evaluation_loader: Use a different dataloader to test the train set (if you want to subsample to take less time)
    :return: Evaluator
    """
    assert epoch > 0, "Epoch should start at 1"

    # Variables
    running_loss = 0.0
    iterations = 0

    # Constants
    batch_size = train_loader.batch_size
    total_samples = len(train_loader) * batch_size

    # Helper functions
    rounded = lambda decimal: str(round(decimal*100, 2))
    run_every = lambda i: (iterations % i <= batch_size) and (iterations > batch_size)
    global_iterations = lambda: iterations + (epoch - 1) * total_samples
    
    def write_test_evaluator(evaluator):
        write_evaluator(writer, "test", evaluator, global_iterations())
        write_histogram(writer, net, global_iterations())
        if checkpoint_path:
            name = f"iterations-{global_iterations()}" + \
                   '-total-' + rounded(evaluator.total_percent_correct())+ \
                   '-class0-' + rounded(evaluator.percent_correct(0)) + \
                   '-class1-' + rounded(evaluator.percent_correct(1)) + \
                   '.pt'
            path = os.path.join(checkpoint_path, name)
            print(f'Writing model: {os.path.basename(path)}')
            save_checkpoint(path, name, net, optimizer, loss)

    gen_batches = train_batches(train_loader, net, optimizer, criterion)
    for batch_num, stats in enumerate(gen_batches, 1):
        loss = stats['loss']
        running_loss += loss
        iterations += batch_size

        if run_every(print_loss_every):
            print_loss(batch_num, loss, epoch, train_loader)
            if write: write_loss(writer, loss, global_iterations())

        if run_every(print_test_evaluation_every):
            evaluator = evaluation(net, test_loader, "test loader")
            print()
            if write: write_test_evaluator(evaluator)

        if run_every(yield_every):
            evaluator = evaluation(net, test_loader, "test loader")
            print_evaluation(evaluator, "test")
            yield evaluator

        if run_every(print_train_evaluation_every):
            loader = train_evaluation_loader or train_loader
            evaluator = evaluation(net, loader, "train loader")
            if write: write_evaluator(writer, "train", evaluator, global_iterations())


    # Print test evaluation at end of epoch
    if evaluate_at_end:
        evaluator = evaluation(net, test_loader, "test loader")
        yield evaluator
    else:
        yield


@wraps(train_epoch)
def train(*args, **kwargs):
    """
    yield_every should be set to infinity
    """
    for evaluator in train_epoch(*args, **kwargs):
        pass

    return evaluator

@wraps(train_epoch)
def train_best_model(epochs, *args, **kwargs):
    best = None
    best_epoch = 1
    for epoch in range(epochs):
        for evaluator in train_epoch(epoch+1, *args, **kwargs):
            if best is None:
                best = evaluator
            elif evaluator.normalized_percent_correct(weigh_events=1.1) >= best.normalized_percent_correct(weigh_events=1.1):
                best = evaluator
                best_epoch = epoch
    return best, best_epoch


@wraps(train_best_model)
def train_sample_sizes(hyper_params, train_loader, test_loader, final_loader, net, optimizer, criterion, copy_loaders=True, train_split=.8, subsample_ratio={0: 1, 1: 1}, csv_write_path=None, **kwargs):
    """
    :hyper_params: should be a list of list, each sublist containing the samples and epochs. Example:
                   samples = [10,   10,   50,   100,  200]
                   epochs =  [100, 100,  100,  50,   50]
                   hyper_params = zip(samples, epochs)

    Have to pass parameters to train_best_model as **kwargs
    """

    if not (0 <= train_split <= 1):
        raise ValueError("train_split should be between 0 and 1")
        
    results = []
    _net, _optimizer, _criterion = net, optimizer, criterion

    if copy_loaders:
        train_loader, test_loader = copy.deepcopy(train_loader), copy.deepcopy(test_loader)

    dataset_train = train_loader.dataset
    dataset_test = train_loader.dataset

    for samples, epochs in hyper_params:
        net = copy.deepcopy(_net)
        criterion = copy.deepcopy(_criterion)

        # In case frozen parameters: https://github.com/amdegroot/ssd.pytorch/issues/109
        optimizer = _optimizer.__class__(filter(lambda p: p.requires_grad, net.parameters()), **optimizer.defaults)
        optimizer.load_state_dict(_optimizer.state_dict())

        _train_dataset = subsample_dataset(dataset_train, math.floor(samples * train_split), subsample_ratio)
        _test_dataset = subsample_dataset(dataset_test, math.floor(samples * (1 - train_split)), subsample_ratio)

        replace_loader_dataset(train_loader, _train_dataset)
        replace_loader_dataset(test_loader, _test_dataset)

        evaluator, best_epoch = train_best_model(
                epochs=epochs,
                train_loader=train_loader, 
                test_loader=test_loader, 
                net=net,
                optimizer=optimizer,
                criterion=criterion,
                **kwargs)

        final_evaluator = evaluation(net, final_loader, "Final Test", copy_net=True)
        results.append( 
            (samples, epochs, (best_epoch, evaluator, final_evaluator)) 
            ) # epoch, validation results, final results

    if csv_write_path:
        for (sample_size, epochs, data) in results:
            final_eval = data[-1]
            csv_write.write_evaluator(final_eval, csv_write_path, extra_row_data=[sample_size, epochs], extra_header_data=['Samples', 'Epochs'])

    return results

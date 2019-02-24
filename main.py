import matplotlib
matplotlib.use('agg')
matplotlib.interactive(False)

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loaders.multiple_loader import SpectrogramMultipleDataset
from loaders.custom_path_loader import SpectrogramCustomPathDataset
from loaders.direct_loader import SpectrogramDirectDataset
from loaders.unknown_loader import SpectrogramUnknownDataset
import models
import os
from datetime import datetime
import config
from evaluator.evaluator import Evaluator
from writer_util import MySummaryWriter as SummaryWriter
from utils import dotdict, verify_dataset_integrity
import sys
import utils
from pprint import pprint
from evaluator.csv_write import write_unknown_predictions_to_csv
from writer_util.stats_writer import StatsWriter

configuration = 'benz_train_set_old'
settings = config.options[configuration]
print(f"Using config {configuration}")

settings = dotdict(settings)
CWD = os.getcwd()

# Train and test
make_path = lambda path: os.path.join(CWD, os.path.join('data', path))

TRAIN_IMG_PATH = make_path(settings.train.path)
TEST_IMG_PATH = make_path(settings.test.path)

# Variables
BATCH_SIZE = 512   # 128
NUM_CLASSES = 2
iterations = 0

WEIGH_CLASSES = settings.weigh_classes

# Neural Net Model
NET = models.mnist_three_component
MODEL_PATH = f'checkpoints/{NET.__name__}'

# Visualize
path = os.path.join(os.path.join(config.VISUALIZE_PATH, f'runs/{NET.__name__}/trial-{datetime.now()}'))
writer = SummaryWriter(path)

# Dimentional Transforms
height_percent, width_percent = settings.image.crop or (1, 1)  # 0.5 to 1.0
height, width = settings.image.height, settings.image.width

resize = (height, width)
crop = (int(height * height_percent), int(width * width_percent)) 
    
crop_padding = settings.image.padding
if crop_padding:
    crop_padding = utils.calculate_crop_padding_pixels(crop_padding, height, width)

# DATASET

Dataset =  SpectrogramDirectDataset if settings.loader == "direct" else SpectrogramCustomPathDataset  # SpectrogramUnknownDataset

dataset_args = dict(
    path_pattern=settings.path_pattern or '',
    crop=crop,
    resize=resize
    )

pprint(settings)

dataset_train = Dataset(img_path=TRAIN_IMG_PATH,
                        transform=NET.transformations['train'],
                        ignore=settings.train.get('ignore'),
                        divide_test=settings.train.divide_test,
                        crop_padding = crop_padding,
                        **dataset_args
                        )

dataset_test = Dataset(img_path=TEST_IMG_PATH,
                        transform=NET.transformations['test'],
                        ignore=settings.test.get('ignore'),
                        divide_test=settings.test.divide_test,
                        test=True,
                        #crop_center=True,
                        **dataset_args
                        )

assert verify_dataset_integrity(dataset_train, dataset_test)

train_sampler = utils.make_weighted_sampler(dataset_train, NUM_CLASSES, weigh_classes=WEIGH_CLASSES) if WEIGH_CLASSES else None

# Data Loaders
loader_args = dict(
                   batch_size=BATCH_SIZE,
                   num_workers=8,
                   pin_memory=True,
                   drop_last=True
                   )

train_loader = DataLoader(dataset_train,
                          shuffle=not train_sampler,
                          sampler=train_sampler,
                          **loader_args)

train_test_loader = DataLoader(dataset_train,
                               **loader_args)

test_loader = DataLoader(dataset_test,
                         **loader_args)


# Setup Net
net = NET().cuda()
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss().cuda()


def guess_labels(batches):
    """
    Tries to guess labels based on data
    """
    dataiter = iter(test_loader)

    for i in range(batches):
        images, labels = dataiter.next()
        images = [Variable(image).cuda() for image in images]
        print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(BATCH_SIZE)))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted:   ', ' '.join('%5s' % predicted[j] for j in range(BATCH_SIZE)))
        print()


def evaluate(net, data_loader, copy_net=False):
    """
    :param net: neural net
    :param copy_net: boolean
    :param data_loader: DataLoader
    :return: Data structure of class Evaluator containing the amount correct for each class
    """
    if copy_net:
        Net = NET().cuda()
        Net.load_state_dict(net.state_dict())
        Net.eval()
    else:
        Net = net

    i = 0
    all_output_labels = torch.FloatTensor()
    all_true_labels = torch.LongTensor()

    for (inputs, labels) in data_loader:
        inputs, labels = [Variable(input).cuda() for input in inputs], labels
        output_labels = Net(inputs).cpu().detach()

        all_output_labels = torch.cat((all_output_labels, output_labels))
        all_true_labels = torch.cat((all_true_labels, labels))

        i += BATCH_SIZE
        sys.stdout.write('\r' + str(i) + '/' + str(BATCH_SIZE * len(data_loader)))
        sys.stdout.flush()


    # Update the information in the Evaluator
    eval = Evaluator(all_true_labels, all_output_labels, 2)

    return eval


def save_model(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_model(path):
    return net.load_state_dict(torch.load(path))


def print_evaluation(evaluator, description):
    correct = 100 * evaluator.total_percent_correct()
    print('Accuracy of the network on %s images: %d %%' % (description, correct))

    for name, info in evaluator.class_info.items():
        print('Accuracy of the network on %s class  %s: [%2d / %2d] %2d %%' % (description, name,
                                                                               info.amount_correct,
                                                                               info.amount_total,
                                                                               evaluator.percent_correct(name) * 100))


def test(net, loader, copy_net=False):
    """
    :param net: the net to test on
    :param copy_net: boolean, whether to copy the net (if in the middle of training, you won't want to use the current net)
    """
    evaluator = evaluate(net, data_loader=loader, copy_net=copy_net)
    return evaluator


def write_images():
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


def write_histogram(net, n):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n)


def write_info():
    transforms = ['Resize: ' + str(resize), 'Crop: ' + str(crop)]

    step = 0
    for transformation in transforms + NET._train + NET._transformations:
        writer.add_text('Transformations_Train', str(transformation), global_step=step)
        step += 1

    step = 0
    for transformation in transforms + NET._test + NET._transformations:
        writer.add_text('Transformations_Test', str(transformation), global_step=step)
        step += 1

    writer.add_text('Configuration', str(settings))

def train(epoch, write=True, yield_evaluator=False):
    global iterations
    running_loss = 0.0
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


        running_loss += loss.data.item()

        def print_loss():
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch,
                  int(i * len(true_inputs) * BATCH_SIZE / 3),
                  len(train_loader) * BATCH_SIZE,
                  100. * i / len(train_loader),
                  loss.data.item())

            sys.stdout.write('\r' + msg); sys.stdout.flush()

        def write_loss():
            writer.add_scalar('train_loss', loss.data.item(), iterations)

        percent_correct = lambda evaluator: evaluator.total_percent_correct() * 100

        def test_loss():
            print("\nTesting...")
            test_evaluator = evaluate(net, test_loader, copy_net=True); print()

            print_evaluation(test_evaluator, 'test')

            writer.add_scalars('amount_correct',
                               {'test_amount_correct': percent_correct(test_evaluator)},
                               iterations)

            writer.add_scalars('test_class_correct',
                               {'test_noise': test_evaluator.percent_correct(0),
                                'test_local': test_evaluator.percent_correct(1)},
                               iterations)
                               
            return test_evaluator

        def train_loss():
            train_evaluator = evaluate(net, train_test_loader, copy_net=True); print()
            print_evaluation(train_evaluator, 'train'); print()
            writer.add_scalars('amount_correct',
                               {'train_amount_correct': percent_correct(train_evaluator)},
                               iterations)

        def write_model(name):
            path = f'./checkpoints/{NET.__name__}/model-{name}.pt'
            print(f'Writing model: {name}')
            save_model(path)

        iterations += BATCH_SIZE

        if i == 0:
            print()

        if iterations % 1000 < BATCH_SIZE:
            print_loss()


        if iterations % 1000 < BATCH_SIZE:
            write_loss()

        if iterations % 5000 < BATCH_SIZE:
            rounded = lambda decimal: str(round(decimal, 4))
            evaluator = test_loss()
            write_model(
                        str(iterations // 5000) +  \
                        '-' + rounded(evaluator.total_percent_correct()) + \
                        '-' + rounded(evaluator.percent_correct(0)) + \
                        '-' + rounded(evaluator.percent_correct(1)) \
                        )
            print()

        if epoch % 2 == 0 and epoch >= 2 and i == 0:
            train_loss()

        # Don't run the write conditions if set to false
        if write:
            if iterations % 10000 < BATCH_SIZE:
                write_histogram(net, iterations)




if __name__ == '__main__':


    ################
    #
    # TRAINING MODEL
    #################

    # print("\nWriting Info")
    # write_info()
    # write_images()
    #
    # def train_net(epochs):
    #     for epoch in range(epochs):
    #         train(epoch)
    #
    # train_net(50)

    ########################
    #RUN MODEL
    #######################
    #

    # Load Net
    MODEL_TO_LOAD = 'BestModel.pt'

    def load_net():
        path = f'./checkpoints/{NET.__name__}/{MODEL_TO_LOAD}'
        load_model(path)
        net.eval()

    print("Loading Net")
    load_net()

    # Test the evaluator
    print("Testing Net")
    test_evaluator = evaluate(net, test_loader, copy_net=True)
    print()
    print_evaluation(test_evaluator, 'test')

    # Write figures
    print("Writing stats...")
    # stats_writer = StatsWriter(os.path.join(CWD, 'visualize/test_stats'))
    # stats_writer.write_stats(test_evaluator.true_labels, test_evaluator.output_labels, test_evaluator.predicted_labels)
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    fig = skplt.metrics.plot_precision_recall(test_evaluator.true_labels, test_evaluator.output_labels)
    plt.savefig('visualize/pr.png')

    # Write CSV predictions
    write_unknown_predictions_to_csv(net, test_loader, 'evaluator/predictions.csv')
    print("\nWrote csv")
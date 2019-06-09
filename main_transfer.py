from pytorch_utils.utils import evaluate, write_images, load_model, print_evaluation, write_info, train
import models
import copy
import torch.nn as nn
from main import *

NET = models.mnist_three_component_exp

 # Load Net
model_path = 'everywhere-97percent-nobenz'
model_name = "iterations-6778880-total-97.73-class0-97.3-class1-98.49.pt"
CHECKPOINT_PATH = f'./visualize/saved/2019/{NET.__name__}/{model_path}/checkpoints/{model_name}'

# Use existing model
print("Loading Net")
net = NET().cuda()

checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']
print("Loaded", checkpoint['name'])
net.train()

def replace_model(net):
    net.classifier = nn.Sequential(
        nn.Linear(4800, 128), 
        nn.ReLU(), 
        nn.Dropout(0.4),
        nn.Linear(128, 2))

# Freeze convolutional parameters
def freeze_parameters(net):
    for param in net.feats.parameters():
        param.requires_grad = False

# NEW
replace_model(net)
net.cuda()
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss().cuda()

# Set up loaders
ratio = {0: 3, 1: 1}
data_train = subsample_dataset(dataset_train, 100, ratio) # subsample_dataset
data_test = reduce_dataset(dataset_test, 5000)

train_sampler = utils.make_weighted_sampler(data_train, NUM_CLASSES, weigh_classes=[1, 1])

train_loader = DataLoader(data_train,
                          shuffle=not train_sampler,
                          sampler=train_sampler,
                          **loader_args)

test_loader = DataLoader(data_test,
                         **loader_args)

# Train on new data
def train_net(epochs):
    for epoch in range(epochs):
        train(epoch + 1, train_loader, test_loader, optimizer, criterion, net, writer,
              write=True,
              checkpoint_path=checkpoint_path,
              print_loss_every=1000,
              print_test_evaluation_every=10000)



# Initial Results before Transfer Learning
print("Testing Net -- Initial")
test_evaluator = evaluate(net, test_loader, copy_net=True)
print()
print_evaluation(test_evaluator, 'test')

# Train it
train_net(200)

    


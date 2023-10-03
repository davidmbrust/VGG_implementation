import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import argparse
import numpy as np
import random as rand
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

#from ..transforms._presets import ImageClassification
#from ..utils import _log_api_usage_once
#from ._api import register_model, Weights, WeightsEnum
#from ._meta import _IMAGENET_CATEGORIES
#from ._utils import _ovewrite_named_param, handle_legacy_interface

# training an Epoch Function
# change to not use global variables



# after epoch 50, stopped, restarted with eight decay 0.005
# and with batch size 128 
# change this back and run from las in VGGModelStates to continue



class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def set_optimizer(learn_r, momen=0.9, weight_d=0.0005):
    # Choosing Optimizer and Loss Function, default lr=0.001, momentum=0.9
    optim_internal = torch.optim.SGD(model.parameters(), lr=learn_r, momentum=momen, weight_decay= weight_d)
    return optim_internal

def train_one_epoch(epoch_index, train_loader, input_model, optim, loss_func, prof):#, Q):#, prof):
    start_time = time.time()
    running_loss = 0.
    last_loss = 0.
    time_1000 = 0.
    step_time = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # pointers not loading, still uses cpu to load up images
    # loading up images is single threaded
    # optimize image loading
    # use profilier ot see if gpu bound
    prof.start()
    for i, data in enumerate(train_loader):
        #if i == 0:
        #    prof.start()
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients every batch, set_to_none=True reduces time
        optim.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = input_model(inputs)

        # Compute the loss and its gradients
        loss_start = time.time()
        loss = loss_func(outputs, labels)
        loss.backward()
        optim_start = time.time()
        loss_time = optim_start - loss_start
        #print('Time for loss function:',loss_time)
        # Adjust learning weights
        optim.step()
        optim_stop = time.time()

        step_time += optim_stop - optim_start
        #print('Time for optimizer:',step_time)
        # Gather data and report
        # save weight matrix, turn off, run again and compare, are they the same
        # also compare a few batches later
        # check to make sure
        # accre??
        running_loss += loss.item()
        #prof.step()
        #prof.stop()
        #print(prof.key_averages())
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} || loss: {} || Timestamp: {}'.format(i + 1, last_loss, datetime.now()))
            #tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            #if i == 999:
                #prof.stop()
                #print(prof.key_averages())
    scheduler.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    #prof.stop()
    #print(prof.key_averages())
    #Q.put(last_loss, epoch_time, step_time)
    return last_loss, epoch_time, step_time



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train VGG16 Network"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=128,
        help="batch size for the images passing through networks",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        help="directory of the images data",
        default="/data/ImageNet/",
    )
    parser.add_argument(
        "--img_trans_train",
        "-itr",
        type=str,
        help="functions desired for training transforms",
        default="ToTensor, Resize-256, RandomCrop-224"
    )
    parser.add_argument(
        "--img_trans_test",
        "-ite",
        type=str,
        help="functions desired for testing transforms",
        default="ToTensor, Resize-256, CenterCrop-224"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        help="initial learning rate for training",
        default=0.01
    )
    parser.add_argument(
        "--momentum",
        "-mom",
        type=float,
        help="momentum for training",
        default=0.9
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="momentum for training",
        default=0
    )
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        help="number of epochs to run",
        default=74
    )
    parser.add_argument(
        "--from_trained",
        "-ft",
        type=str,
        help="model to train from"
    )
    args = parser.parse_args()

    # on restart after epoch 25, lr=1.0000000000000002e-08

    # use_deterministic makes all operations deterministic, not just backends behavior
    torch.use_deterministic_algorithms(mode=True)

    # causes cuDNN to deterministically select an algorithm
    torch.backends.cudnn.benchmark = False

    # setting seeds for types of functions
    random_seed = args.seed
    torch.manual_seed(random_seed)
    rand.seed(random_seed)
    np.random.seed(random_seed)

    # Creating Image Transformations

    # Cropping and seeding for randomization / image order
    # GPU variability harder to control
    # getting GPU determinabiolity in pytorch (harder)

    # make flags for implementation (maybe turn off initially)
    '''''
    trans_list_train = args.img_trans_train.split(', ')
    for trans in trans_list_train:
        trans.replace(' ', '')
    trans_train = ['torchvision.transforms.' + x for x in trans_list_train]


 
    trans_list_test = args.img_trans_test.split(', ')
    for trans in trans_list_test:
        trans.replace(' ', '')
    print('\n\n\n')
    print(trans_list_test)
    print('\n\n\n')

    
    transforms_training = exec('torchvision.transforms.Compose([' + ', '.join(trans_train) + '])')
    transforms_testing = exec('torchvision.transforms.Compose([' + ', '.join(trans_test) + '])')
    print('\n\n\n')
    print(transforms_testing)
    print('\n\n\n')
    '''''
# run through several epochs with profiler
# fancy pca color shift is what paper uses
    transforms_training = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.ColorJitter(),
    #torchvision.transforms.RandomAffine(),
    #torchvision.transforms.RandomPerspective(),  
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.RandomCrop((224,224)), 
    #torchvision.transforms.RandomVerticalFlip(), 
    #torchvision.transforms.RandomHorizontalFlip(),
    ])

    transforms_testing = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.CenterCrop((224,224))
    ])

    # Creating Datasets
    # try number of workers or device to cuda
    # prefetching!!, etc
    # working on future batches as GPU runs
    # leverage parallelizing 
    # want to see accuracy
    # similar accuracy as in paper
    # two types, exactly correct or within top 5 guesses
    # need ot make it to go much faster


    # same or valid average pooling
    # see if truly deterministic



    training_dataset = datasets.ImageFolder(root= args.data_dir + 'train', transform= transforms_training)
    test_dataset = datasets.ImageFolder(root= args.data_dir + 'sortedVal', transform= transforms_testing)

    # Creating Data Loaders
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 14, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 14)

    # Choosing device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializing epoch number
    epoch_number = 0

    # initializing VGG16 model
    model = VGG16()
    # using GPUs
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)

    torch.autocast('cpu', enabled=False)
    # lr=0.001, momentum=0.9
    # lr=0.0001 after 19 epochs
    # lr=0.00001 after 29

    learning_rate = args.learning_rate

    # Chosing Optimier and Loss Function
    optimizer = set_optimizer(learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    #torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay= 0.005)

    # make sure the cross entropy is same for binary / categorical
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    #torch.use_autocast = False
    # Loading VGG model 
    if args.from_trained:
        print('Using Pretrained Model State')
        pretrain_name = args.from_trained
        model.load_state_dict(torch.load(pretrain_name))
        epoch_idx = pretrain_name.find('_ep-')
        print('\n\n\nEpoch idx:', epoch_idx)
        print(pretrain_name[epoch_idx+4:epoch_idx+6])
        epoch_number = int(pretrain_name[epoch_idx+4:epoch_idx+6]) + 1
        print('\nEpoch Number:', epoch_number)


    # Initializing epochs, vloss and timimg
    EPOCHS = args.num_epochs - epoch_number
    print('Total Epochs Remaining:', EPOCHS)
    timestamp = datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
    i = 0
    argStr= []
    prev_accuracy=0
    for key, value in vars(args).items():
    # do stuff
        argStr.append(f'ag-{str(key)}<{str(value)}')
    '_'.join(argStr)
    print(argStr)
    # inject information about paramerters, make title all the parameters
    # or can write a header into the file, concatenate afterward
    # make sure you are saving exact params into file
    writer = SummaryWriter('runs/vgg16-ImageNet_trainer_ts-{}_{}'.format(timestamp, argStr))
    best_vloss = 1_000_000.

    # Running loop for training
    prof = torch.profiler.profile(
           activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof_logs/VGG16'),
            #record_shapes=True,
            #profile_memory=True,
            #with_stack=True,
            #with_flops=True,
            #with_modules=True
            )
    for epoch in range(EPOCHS):
        total_start_time = datetime.now()
        print('EPOCH {} | {}:'.format(epoch_number + 1, datetime.now()))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        #model.share_memory()
        train_processes = []
        num_processes = 16
        #set_start_method('spawn')
        #Que = mp.Queue()
        #for rank in range(num_processes):
        #p = mp.spawn(train_one_epoch, args=(epoch_number, training_loader, model, optimizer, loss_fn, Que))
        avg_loss, epoch_time, step_time = train_one_epoch(epoch_number, training_loader, model, optimizer, loss_fn, prof)
        #p.start()
            #train_processes.append(p)
        #sfor p in train_processes:
        #p.join()
        #avg_loss, epoch_time, step_time = Que.get()
        running_vloss = 0.0
        top1_correct_total = 0.0
        top5_correct_total = 0.0
        total_samples = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        # want to know about model.eval()
        # diabling dropout? == Yes
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                top1_correct_total += (voutputs.argmax(1) == vlabels).sum().item()
                top5pred = torch.topk(voutputs, 5)
                for pred in top5pred:
                    top5_correct_total += (pred == vlabels).sum().item()
                total_samples += vlabels.size(0)
        avg_vloss = running_vloss / (i + 1)
        accuracy1 = top1_correct_total / total_samples
        accuracy5 = top5_correct_total / total_samples
        print('LOSS: train {} valid {}'.format(avg_loss, avg_vloss))
        print('Top1 Accuracy:', accuracy1)
        print('Top5 Accuracy:', accuracy5)
        '''''
        if (accuracy1 <= prev_accuracy):
            learning_rate = learning_rate / 10
            optimizer = set_optimizer(learning_rate)
            print(f'-------------\nUpdating Optimizer to use lr={learning_rate}\n-------------')
        '''''
        # Log the running loss averaged per batch
        # for both training and validation
        try:
            current_lr = scheduler.getlr()
        except:
            current_lr = 0

        writer.add_scalars('Training , Validation Loss, Learning Rate, Accuracies',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss, 'LearningRate' : current_lr, 'AccuracyTop1' : accuracy1, 'AccuracyTop5' : accuracy5, 'Epoch Time': epoch_time},
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        # figure out how big model is
        # is it viable to save every model??
        # just save the weights?
        # want to be able to track performance over epoch / time
        # saving epoch is useful in case computer crashes
        # saving just weights wouldnt retain state of training
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            vloss_better = True
        else:
            vloss_better = False
        timestamp = datetime.now()
        total_end_time = datetime.now()
        total_time = total_start_time - total_end_time
        model_path = '/data/brustdm/model_imp/VGG16ModelStates(Newer)/model_ts-{}_ep-{}_et-{}_tt-{}_st-{}_bv-{}_a1-{}_a5-{}'.format(timestamp, epoch_number, epoch_time, total_time, step_time, vloss_better, accuracy1, accuracy5)
        torch.save(model.state_dict(), model_path)
        prev_accuracy = accuracy1
        epoch_number += 1


    # note how long training step takes and epoch takes
    # ballpark it should take around a week to train
    # 3 seconds per step over time is quite a lot
    # mem/ cpu/ gpu, maximize use of all 3 at all times
    # must have bottleneck but get the best as possible
    # small models usually cpu bottleneck
    # possible to use both GPUs to train one model


    # definitely look at accuracy, what matters for replication

    # pytorch profiling tools, blocks that show bottlenecks (tensorflow version)
    # same, one batch , one step, one epoch
    # experimental new vscode, makes notebooks faster
    # notebooks better for presentation only

    # copilot?


    # implement topk!!!! 
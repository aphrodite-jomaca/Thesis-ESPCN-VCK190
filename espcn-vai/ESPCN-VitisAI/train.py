'''
Simple PyTorch MNIST example - training & testing
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys
import os
import shutil

from common import *
from data import get_training_set, get_test_set


DIVIDER = '-----------------------------------------'



def train_test(build_dir, upscale_factor, trainbatchsize, testbatchsize, learnrate, epochs):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'

    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    model = Net(upscale_factor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnrate)


    #image datasets
    train_set = get_training_set(upscale_factor)
    test_set = get_test_set(upscale_factor)

    #data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                            batch_size=trainbatchsize, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                            batch_size=testbatchsize, 
                                            shuffle=False)


    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


    # save the trained model
    shutil.rmtree(float_model, ignore_errors=True)    
    os.makedirs(float_model)   
    save_path = os.path.join(float_model, 'f_model.pth')
    torch.save(model.state_dict(), save_path) 
    print('Trained model written to',save_path)

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d',       '--build_dir',        type=str,   default='build',    help='Path to build folder. Default is build')
    ap.add_argument('-s',       '--upscale_factor',   type=int,   default=2,          help='Upscaling factor. Must be an integer. Default is 2')
    ap.add_argument('-train_b', '--trainbatchsize',   type=int,   default=4,          help='Training batchsize. Must be an integer. Default is 4')
    ap.add_argument('-test_b',  '--testbatchsize',    type=int,   default=50,         help='Testing batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e',       '--epochs',           type=int,   default=350,        help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr',      '--learnrate',        type=float, default=0.001,      help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir        : ',args.build_dir)
    print ('--upscale_factor   : ',args.upscale_factor)
    print ('--trainbatchsize   : ',args.trainbatchsize)
    print ('--testbatchsize    : ',args.testbatchsize)
    print ('--learnrate        : ',args.learnrate)
    print ('--epochs           : ',args.epochs)
    print(DIVIDER)

    train_test(args.build_dir, args.upscale_factor, args.trainbatchsize, args.testbatchsize, args.learnrate, args.epochs)

    return



if __name__ == '__main__':
    run_main()

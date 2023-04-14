'''
Make the target folder
Copies images, application code and compiled xmodel to 'target'
'''

'''
Author: Mark Harvey
'''

import torch
import torchvision

import argparse
import os
import shutil
import sys
import cv2
import numpy as np
from tqdm import tqdm

from data import get_test_set


DIVIDER = '-----------------------------------------'

def generate_images(upscale_factor, num_images, lr_dir, hr_dir):

  # BDSD300 test dataset and dataloader
  test_set = get_test_set(upscale_factor)

  test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                            batch_size=1, 
                                            shuffle=False)

  # iterate thru' the dataset and create images
  dataiter = iter(test_loader)
  for i in tqdm(range(num_images)): #num_images
    lr, hr = dataiter.next()
    # lr image
    lr_img = lr.numpy().squeeze()
    lr_img = (lr_img * 255.)#.astype(np.uint8)
    lr_img_file=os.path.join(lr_dir, 'test_'+str(i)+'.png')
    cv2.imwrite(lr_img_file, lr_img)
    # hr image
    hr_img = hr.numpy().squeeze()
    hr_img = (hr_img * 255.)#.astype(np.uint8)
    hr_img_file=os.path.join(hr_dir, 'test_'+str(i)+'.png')
    cv2.imwrite(hr_img_file, hr_img)

  return


def make_target(build_dir,target,upscale_factor,num_images,app_dir):

    # dset_dir = build_dir + '/dataset'
    comp_dir = build_dir + '/compiled_model'
    target_dir = build_dir + '/target_' + target

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app.py'), target_dir)
    shutil.copy(os.path.join(app_dir, 'imgproc.py'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/ESPCN_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)

    # create lr, hr images
    lr_dir = target_dir + '/lr_image_dir'
    shutil.rmtree(lr_dir, ignore_errors=True)
    os.makedirs(lr_dir)
    hr_dir = target_dir + '/hr_image_dir'
    shutil.rmtree(hr_dir, ignore_errors=True)
    os.makedirs(hr_dir)

    generate_images(upscale_factor, num_images, lr_dir, hr_dir)
    
    # create sr images' dir
    sr_dir = target_dir + '/sr_image_dir'
    shutil.rmtree(sr_dir, ignore_errors=True)
    os.makedirs(sr_dir)

    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',        type=str,  default='build',         help='Path to build folder. Default is build')
    ap.add_argument('-t', '--target',           type=str,  default='vck190',        choices=['zcu102','zcu104','u50','vck190'], help='Target board type (zcu102,zcu104,u50,vck190). Default is zcu102')
    ap.add_argument('-s', '--upscale_factor',   type=int,  default=2,               help='Upscaling factor. Must be an integer. Default is 2')
    ap.add_argument('-n', '--num_images',       type=int,  default=1,             help='Number of test images. Default is 1')
    ap.add_argument('-a', '--app_dir',          type=str,  default='application',   help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir      : ', args.build_dir)
    print (' --target         : ', args.target)
    print (' --upscale_factor : ', args.upscale_factor)
    print (' --num_images     : ', args.num_images)
    print (' --app_dir        : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir, args.target, args.upscale_factor, args.num_images, args.app_dir)


if __name__ ==  "__main__":
    main()

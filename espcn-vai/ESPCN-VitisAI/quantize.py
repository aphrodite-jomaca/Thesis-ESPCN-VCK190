'''
Simple PyTorch MNIST example - quantization
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from common import *
from data import get_training_set, get_test_set


DIVIDER = '-----------------------------------------'




def quantize(build_dir,quant_mode,inspect,upscale_factor,batchsize):

  dset_dir = build_dir + '/dataset'
  float_model = build_dir + '/float_model'
  quant_model = build_dir + '/quant_model'


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

  # load trained model
  model = Net(upscale_factor).to(device)
  model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))

  rand_in = torch.randn([batchsize, 1, 16, 16])

  if inspect:
    import sys
    from pytorch_nndct.apis import Inspector

    inspector = Inspector("DPUCVDX8G_ISA3_C32B6")

    inspector.inspect(model, (rand_in,), device=device, output_dir="inspect", image_format="png")


  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model


  # data loader
  test_set = get_test_set(upscale_factor)

  test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                            batch_size=batchsize, 
                                            shuffle=False)

  # evaluate 
  test(quantized_model, device, test_loader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',       type=str,  default='build',   help='Path to build folder. Default is build')
  ap.add_argument('-s',  '--upscale_factor',  type=int,  default=2,         help='Upscaling factor. Must be an integer. Default is 2')
  ap.add_argument('-q',  '--quant_mode',      type=str,  default='calib',   choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-i',  '--inspector',       type=bool, default=True,     help='Activate Inspector. Default is False')
  ap.add_argument('-b',  '--batchsize',       type=int,  default=1,       help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir       : ',args.build_dir)
  print ('--upscale_factor  : ',args.upscale_factor)
  print ('--quant_mode      : ',args.quant_mode)
  print ('--inspector       : ',args.inspector)
  print ('--batchsize       : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.inspector,args.upscale_factor,args.batchsize)

  return



if __name__ == '__main__':
    run_main()

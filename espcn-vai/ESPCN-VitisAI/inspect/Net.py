# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Net::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #Net::Net/Conv2d[conv1]/168
        self.module_2 = py_nndct.nn.Tanh() #Net::Net/Tanh[tanh]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/Conv2d[conv3]/188
        self.module_4 = py_nndct.nn.Tanh() #Net::Net/Tanh[tanh]/input
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=4, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/Conv2d[conv4]/208
        self.module_6 = py_nndct.nn.Module('pixel_shuffle',upscale_factor=2) #Net::Net/PixelShuffle[pixel_shuffle]/210

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        return output_module_0

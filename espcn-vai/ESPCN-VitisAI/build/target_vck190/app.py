from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

import imgproc

_divider = '-------------------------------'


def preprocess_fn(lr_image_path, hr_image_path):
    '''
    Image pre-processing.
    Opens image as grayscale, adds channel dimension, normalizes to range 0:1
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''
    # Read LR image and HR image    
    lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.0
    hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0

    # Convert BGR image to YCbCr image
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
    hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

    # Split YCbCr image data
    lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
    hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

    return lr_y_image, hr_y_image, hr_cb_image, hr_cr_image

def postprocess_fn(lr_image_path, hr_image_path, qscale):
    '''
    Image post-processing.
    '''

def Tanh(xx):
    x = np.asarray( xx, dtype="float32" )
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()]
    output_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)


def runDPU(dpu_1,dpu_3,dpu_5,img):
    '''get tensor'''
    inputTensor_1 = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors()
    inputTensor_3 = dpu_3.get_input_tensors()
    outputTensor_3 = dpu_3.get_output_tensors()
    inputTensor_5 = dpu_5.get_input_tensors()
    outputTensor_5 = dpu_5.get_output_tensors()

    input_ndim1 = tuple(inputTensor_1[0].dims)
    output_ndim1 = tuple(outputTensor_1[0].dims)
    #print("Input  1 dims: ", input_ndim1)
    #print("Output 1 dims: ", output_ndim1)
    input_ndim3 = tuple(inputTensor_3[0].dims)
    output_ndim3 = tuple(outputTensor_3[0].dims)
    #print("Input  3 dims: ", input_ndim3)
    #print("Output 3 dims: ", output_ndim3)
    input_ndim5 = tuple(inputTensor_5[0].dims)
    output_ndim5 = tuple(outputTensor_5[0].dims)
    #print("Input  5 dims: ", input_ndim5)
    #print("Output 5 dims: ", output_ndim5)

    batchSize = input_ndim1[0]
    #print("Batch size: ", batchSize)
    n_of_images = len(img)
    count = 0
    write_index = 0
    timetotal = 0.0
    
    out1 = np.zeros([batchSize,16,16,64], dtype='float32')
    out3 = np.zeros([batchSize,16,16,32], dtype='float32')
    out5 = np.zeros([batchSize,32,32,1], dtype='float32')

    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare input'''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim1, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim1[1:])

        '''run with batch '''
        time1 = time.time()
        execute_async(dpu_1, {"Net__input_0_fix":inputData[0], "Net__Net_Conv2d_conv1__168_fix": out1})
        
        inp2 = out1.copy()
        out2 = Tanh(inp2)
        
        execute_async(dpu_3, {"Net__Net_Tanh_tanh__input_3_fix":out2, "Net__Net_Conv2d_conv3__188_fix": out3})
        
        inp4 = out3.copy()
        out4 = Tanh(inp4)

        execute_async(dpu_5, {"Net__Net_Tanh_tanh__input_fix":out4, "Net__Net_PixelShuffle_pixel_shuffle__210_fix": out5})
        
        nn_out = out5.copy()
        time2 = time.time()
        timetotal += time2 - time1

        for j in range(runSize):
            out_q[write_index] = nn_out
            write_index += 1
        
        count = count + runSize
    return timetotal



def app(hr_image_dir, lr_image_dir, sr_image_dir, threads,model):

    listimage=os.listdir(lr_image_dir)
    runTotal = len(listimage)
    print('Found',len(listimage),'images - processing',runTotal,'of them')

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    print("Len Subgraphs: ",len(subgraphs))

    dpu_subgraph0 = subgraphs[0]
    dpu_subgraph1 = subgraphs[1]
    dpu_subgraph2 = subgraphs[2]
    dpu_subgraph3 = subgraphs[3]
    dpu_subgraph4 = subgraphs[4]
    dpu_subgraph5 = subgraphs[5]
    dpu_subgraph6 = subgraphs[6]

    print("dpu_subgraph0 = " + dpu_subgraph0.get_name())
    print("dpu_subgraph1 = " + dpu_subgraph1.get_name())
    print("dpu_subgraph2 = " + dpu_subgraph2.get_name())
    print("dpu_subgraph3 = " + dpu_subgraph3.get_name())
    print("dpu_subgraph4 = " + dpu_subgraph4.get_name())
    print("dpu_subgraph5 = " + dpu_subgraph5.get_name())
    print("dpu_subgraph6 = " + dpu_subgraph6.get_name())

    dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")
    dpu_3 = vart.Runner.create_runner(dpu_subgraph3, "run")
    dpu_5 = vart.Runner.create_runner(dpu_subgraph5, "run")
    

    ''' preprocess images '''
    print (_divider)
    print('Pre-processing',runTotal,'images...')
    lr_img = []
    hr_img = []
    for i in range(runTotal):
        lr_image_path = os.path.join(lr_image_dir, listimage[i])
        hr_image_path = os.path.join(hr_image_dir, listimage[i])
        lr_y, hr_y, hr_cb, hr_cr = preprocess_fn(lr_image_path, hr_image_path)
        lr_img.append(lr_y)
        hr_img.append((hr_y, hr_cb, hr_cr))

    '''run threads '''
    print (_divider)
    print('Starting DPU execution...')
    threadAll = []
    start=0
    end = len(lr_img)
    in_q = lr_img[start:end]

    timetotal = runDPU(dpu_1, dpu_3, dpu_5, lr_img)

    fps = float(runTotal / timetotal)
    print (_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    print (_divider)
    total_psnr = 0.0
    psnr = 0.0
    for i in range(runTotal):
        sr_y_image = np.reshape(out_q[i][i%6], (32,32))
        #sr_ycbcr_image = cv2.merge([sr_y_image, hr_img[i][1], hr_img[i][2]])
        #sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        #sr_image_path = os.path.join(sr_image_dir, listimage[i])
        #cv2.imwrite(sr_image_path, sr_image * 255.0)
        psnr = 10. * np.log10(1. / np.mean((sr_y_image - hr_img[i]) ** 2))
        print(f"PSNR: {psnr:4.2f}dB.\n")
        total_psnr += psnr

    print (_divider) 
    print(f"Total PSNR: {total_psnr / runTotal:4.2f}dB.\n")
    print (_divider)

    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-hr_dir', '--hr_image_dir', type=str, default='hr_image_dir',        help='Path to folder of HR images. Default is hr_image_dir')
  ap.add_argument('-lr_dir', '--lr_image_dir', type=str, default='lr_image_dir',        help='Path to folder of LR images. Default is lr_image_dir')
  ap.add_argument('-sr_dir', '--sr_image_dir', type=str, default='sr_image_dir',        help='Path to folder of SR images. Default is sr_image_dir')  
  ap.add_argument('-t',      '--threads',      type=int, default=1,                     help='Number of threads. Default is 1')
  ap.add_argument('-m',      '--model',        type=str, default='ESPCN_vck190.xmodel', help='Path of xmodel. Default is ESPCN.xmodel')
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --hr_image_dir : ', args.hr_image_dir)
  print (' --lr_image_dir : ', args.lr_image_dir)
  print (' --sr_image_dir : ', args.sr_image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.hr_image_dir,args.lr_image_dir,args.sr_image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()

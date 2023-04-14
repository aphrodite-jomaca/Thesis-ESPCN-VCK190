# Utilizing Versal Architecture for Low-Latency Super Resolution Applications
This repository contains the source code for the author's thesis in which she implemented an [ESPCN](https://arxiv.org/abs/1609.05158) network performing Super-Resolution on Xilinx's [Versal VCK190](https://www.xilinx.com/products/boards-and-kits/vck190.html) platform. The complete manuscript descibing the whole design and implementation can be found [here](http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/18494). 
## Contents
1. **espcn-hw**  : A Vitis project conaining the HW implementation of the ESPCN network.
2. **espcn-vai** : All the required code to quantize, compile and run a pretrained ESPCN model using the Vitis-AI stack.
3. **ThesisSWUtils** : A Colab notebook that provides all the required SW utils to train and test an ESPCN model, extract its weights & biases, measure its performance, generate inputs for the HW implementation and check the functionality of the design.

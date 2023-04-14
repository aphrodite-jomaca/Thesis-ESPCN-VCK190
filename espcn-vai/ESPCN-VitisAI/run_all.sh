conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# run training
python -u train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log


# quantize & export quantized model
python -u quantize.py -d ${BUILD} --quant_mode calib -i True 2>&1 | tee ${LOG}/quant_calib.log
python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# compile for target boards
source compile.sh vck190 ${BUILD} ${LOG}

# make target folders
python -u target.py --target vck190 -d ${BUILD} 2>&1 | tee ${LOG}/target_vck190.log

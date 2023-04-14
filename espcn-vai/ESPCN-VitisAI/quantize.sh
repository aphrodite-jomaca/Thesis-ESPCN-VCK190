# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# quantize & export quantized model
python -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
#python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# run training
python -u train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log

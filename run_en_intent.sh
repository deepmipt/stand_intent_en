#!/bin/bash
source ./env/bin/activate &&
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/ &&
nohup python3 intent_en_api.py > ./intent_en.log &

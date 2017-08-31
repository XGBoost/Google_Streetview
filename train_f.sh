#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
python resnet50_finetune.py 2>&1 | tee $LOG
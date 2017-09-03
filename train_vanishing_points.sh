#!/bin/bash
LOG=log/train_vanishing_points-`date +%Y-%m-%d-%H-%M-%S`.log
python resnet50_vanishing_points_prediction.py 2>&1 | tee $LOG
#!/bin/bash

python main.py -n_models 34 --dataset cifar10 --arch_type lenet5
python main.py -n_models 34 --dataset fashionmnist --arch_type lenet5
python main.py -n_models 34 --dataset mnist --arch_type lenet5

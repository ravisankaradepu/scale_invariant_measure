import torch
import torchvision
import numpy as np
import time
import pickle
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_networks.vgg as vgg
from quotient_manifold_tangent_vector_pytorch import QuotientManifoldTangentVector,\
                                                     riemannian_power_method,\
                                                     riemannian_hess_quadratic_form
import argparse
from load_cifar_data import load_cifar_as_array


parser = argparse.ArgumentParser()
parser.add_argument('-n_models',
                    '--num_models',
                    type=int,
                    help='Number of modles')
parser.add_argument('-pdb',
                    '--with_pdb',
                    action='store_true',
                    help='run with python debugger')
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--reinit",action='store_true')

args = parser.parse_args()
if args.with_pdb:
    import pdb
    pdb.set_trace()

train_x, train_y, val_x, val_y, test_x, test_y = load_cifar_as_array('data/cifar-10-batches-py/')
batch_size = [256, 2000]

sharpness = []
losses = []
norms = []
for bs in batch_size:
    for r in range(args.num_models):
        net = vgg.vgg16()
        criterion = nn.CrossEntropyLoss()
        weights_file = "{}/pytorch_networks/saves/{}/{}/".format(os.getcwd(),args.arch_type,args.dataset)
        net = torch.load(weights_file+"{}_model_lt.pth.tar".format(r))
        weights,biases = [],[]
        for i,j in zip(net.named_parameters(), net.parameters()):
            if 'weight' in i[0]:
                weights.append(j.data.cpu().numpy())
            if 'bias' in i[0]:
                biases.append(j.data.cpu().numpy())
        layer_sizes = [w.shape for w in weights] + [b.shape for b in biases]
        v_init = [np.random.normal(size=layer_sizes[i]) for i in range(len(layer_sizes))]

        W_orig = QuotientManifoldTangentVector(layer_sizes)
        W_orig.set_vector(weights+biases)

        n_samples = 100

        t1 = time.time()
        v_res,errs = riemannian_power_method(v_init, 1000, net, criterion, W_orig, train_x[:n_samples], train_y[:n_samples], tol=1e-6)
        sp_norm = riemannian_hess_quadratic_form(v_res, net, criterion, W_orig, train_x[:n_samples], train_y[:n_samples])
        secs1 = time.time() - t1
        print('Measuring sharpness took %.4f seconds' % (secs1))

        if torch.cuda.is_available():
            inputs, labels = Variable(torch.Tensor(train_x[:n_samples]).cuda()), Variable(torch.Tensor(train_y[:n_samples]).type(torch.LongTensor).cuda())
            net = net.cuda()
            criterion = criterion.cuda()
        else:
            inputs, labels = Variable(torch.Tensor(train_x[:n_samples])), Variable(torch.Tensor(train_y[:n_samples]).type(torch.LongTensor))

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sharpness.append(sp_norm/(2+2*loss.data.cpu().numpy()))

import matplotlib.pyplot as plt


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

SIZE = 20

plt.plot(sharpness)
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
plt.xlabel('iterations')
plt.ylabel('sharpness')

f = weights_file+'sharpness.jpeg'
if args.reinit:
    f = weights_file+'reinit_sharpness.jpeg'
plt.savefig(f,dpi=600)

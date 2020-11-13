import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn

from thop import clever_format
from thop import profile


def complexity_bar_graph(per_layer_complexity_data, file_prefix,plot = 0):
    mac_data = []
    params_data = []
    act_data = []
    dp_data = []
    weight_reuse = []
    input_reuse = []

    for compute_layer in per_layer_complexity_data:
        mac_data.append(compute_layer[1])
        params_data.append(compute_layer[2])
        act_data.append(compute_layer[3])
        dp_data.append(compute_layer[4])
        weight_reuse.append(compute_layer[5])
        input_reuse.append(compute_layer[6])
    
    if plot == 1:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(act_data)) - 0.125, act_data, width=0.25, label='activation')
        ax.bar(np.arange(len(act_data)) + 0.125, input_reuse, width=0.25, label='data reuse')
        ax.set_title('number of activation per layer and input reuse factor', fontsize=35)
        ax.tick_params(axis="x", labelsize=35)
        ax.tick_params(axis="y", labelsize=35)
        ax.set_xlabel('layer idx', fontsize=35)
        ax.set_yscale('log')
        ax.grid()
        ax.legend(fontsize=25)
        plt.savefig(file_prefix + 'act.png')

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(params_data)) - 0.125, params_data, width=0.25, label='weights')
        ax.bar(np.arange(len(weight_reuse)) + 0.125, weight_reuse, width=0.25, label='weight reuse')
        ax.set_yscale('log')
        ax.set_title('number of weights per layer and weight reuse factor', fontsize=35)
        ax.tick_params(axis="x", labelsize=35)
        ax.tick_params(axis="y", labelsize=35)
        ax.set_xlabel('layer idx', fontsize=35)
        ax.grid()
        ax.legend(fontsize=25)
        plt.savefig(file_prefix + 'weights.png')

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(dp_data)), dp_data, width=0.5)
        ax.set_yscale('log')
        ax.set_title('number of DP per layer', fontsize=35)
        ax.tick_params(axis="x", labelsize=35)
        ax.tick_params(axis="y", labelsize=35)
        ax.set_xlabel('layer idx', fontsize=35)
        ax.grid()
        plt.savefig(file_prefix + 'dp.png')

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(mac_data)), mac_data, width=0.5)
        ax.set_yscale('log')
        ax.set_title('number of MAC per layer', fontsize=35)
        ax.tick_params(axis="x", labelsize=35)
        ax.tick_params(axis="y", labelsize=35)
        ax.set_xlabel('layer idx', fontsize=35)
        ax.grid()
        plt.savefig(file_prefix + 'mac.png')


def profile_model(model, file_prefix):
    input = torch.randn(1, 3, 224, 224)

    macs, params, num_act, num_dp, per_compute_layer_complexity = profile(model, inputs=(input,))
    storage = clever_format([(num_act + params) * 8 / 8], "%.3f")

    macs, params, num_act, num_dp = clever_format([macs, params, num_act, num_dp], "%.3f")

    print(
        'activations:', num_act,
        'weight:', params,
        'num_dp:', num_dp,
        'macs:', macs
    )

    complexity_bar_graph(per_compute_layer_complexity, file_prefix)



if __name__ == '__main__':
    # model = torchvision.models.resnet18()
    # file_prefix = 'resnet18_'
    # model = torchvision.models.vgg11()
    # file_prefix = 'vgg11_'
    # model = torchvision.models.vgg16()
    # file_prefix = 'vgg16_'
    # model = Net(10)
    # file_prefix='alex_cifar_'
    model = torchvision.models.alexnet()
    file_prefix='alexnet_'
    print("Alexnet:")
    
    profile_model(model, file_prefix)

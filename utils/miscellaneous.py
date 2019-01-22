"""
An utils function for those helper function without specific categorized
"""

import numpy as np
import torch
import os
import sys
import operator

def generate_layer_name_collections(net, model_name=None, quantized_first_last_layer=False):
    """
    This function extract convolution / linear layer name for quantization,
    for example: ('conv1', 'layer1.0.conv1)
    :param net:
    :param model_name:
    :param quantized_first_last_layer:
    :return:
    """
    layer_collection_list = list()

    for layer_name in net.state_dict().keys():

        if 'ResNet' in model_name:
            if not quantized_first_last_layer and \
                    (layer_name.startswith('module.conv1.') or (layer_name.startswith('module.linear')) or
                     layer_name.startswith('module.fc')):
                continue
            elif 'conv1.weight' == layer_name or 'module.conv1.weight' == layer_name:
                layer_collection_list.append(layer_name[0: -7])  # [conv1] or [module.conv1]
            elif 'conv1.weight' in layer_name:  # layer1.0.conv1.weight
                layer_name_prefix = layer_name[0: -12]
                layer_collection_list.append(layer_name_prefix + 'conv1')
            elif 'conv2.weight' in layer_name:
                layer_name_prefix = layer_name[0: -12]
                layer_collection_list.append(layer_name_prefix + 'conv2')
            elif 'conv3.weight' in layer_name:
                layer_name_prefix = layer_name[0: -12]
                layer_collection_list.append(layer_name_prefix + 'conv3')
            elif 'downsample.0.weight' in layer_name:  # layer4.0.downsample.0.weight
                layer_name_prefix = layer_name[0: -8]
                layer_collection_list.append(layer_name_prefix + '0')
            elif 'shortcut.0.weight' in layer_name:  # layer4.0.shortcut.0.weight
                layer_name_prefix = layer_name[0: -8]
                layer_collection_list.append(layer_name_prefix + '0')
            # linear.weight, module.linear.weight, classifier.1.weight
            elif 'linear.weight' in layer_name or 'fc.weight' in layer_name or \
                    ('classifier' in layer_name and 'weight' in layer_name):
                layer_name_prefix = layer_name[0: -7]
                layer_collection_list.append(layer_name_prefix)

        else:
            raise NotImplementedError

    return layer_collection_list


def generate_trainable_parameters(all_trainable_param, quantized_conv_name,
                                  model_name=None, quantized_first_last_layer=False):
    """
        This function generate trainable parameters given specific quantized layer, it follows the rules as:
        1) Include weights except already quantized one
        2) Include all bn, bias parameters

        all_trainable_param: an iterator containing all trainable parameters and their names
        quantized_conv_name:
            'layer1.2.conv2.weight', 'layer2.0.downsample.0'
            'module.layer3.1.conv1.weight'
    """
    assert model_name is not None

    trainable_parameters = []
    trainable_param_names = []
    stop_flag = True  # When stop flag become false, all preceding layers are included

    for layer_name, layer_param in all_trainable_param:
        #####################
        # ResNet-like Model #
        #####################
        if 'ResNet' in model_name:
            # Include first and last layer
            if not quantized_first_last_layer and \
                    (layer_name.startswith('module.conv1') or layer_name.startswith('module.linear') or
                     layer_name.startswith('module.fc')):
                trainable_param_names.append(layer_name)
                trainable_parameters.append(layer_param)
            # Include bn layer
            elif 'bn' in layer_name or 'shortcut.1' in layer_name:
                trainable_param_names.append(layer_name)
                trainable_parameters.append(layer_param)
            # Include all layer when stop flag is False
            elif not stop_flag:
                trainable_param_names.append(layer_name)
                trainable_parameters.append(layer_param)
            # It will not include new layer until we reach the quantized layer
            if layer_name == quantized_conv_name:
                stop_flag = False
        else:
            raise NotImplementedError

    return trainable_parameters, trainable_param_names


def folder_init(root, folder_name_list):
    for folder_name in folder_name_list:
        if not os.path.exists('./%s/%s/' % (root, folder_name)):
            os.makedirs('./%s/%s/' % (root, folder_name))


# def fetch_network(model_name):
#     from models_CIFAR10.resnet_layer_input import *
#     from models_ImageNet.resnet_layer_input import *
#
#     if model_name == 'ResNet20':
#         return resnet20_cifar()

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # from models_CIFAR10.resnet import ResNet18 as NetWork
    # from models_ImageNet.resnet import resnet18 as NetWork
    # from models_ImageNet.alexnet_bn_layer_input import alexnet as NetWork
    from models_ImageNet.alexnet_layer_input import alexnet as NetWork
    # from models_CIFAR10.vgg import VGG as NetWork
    # from models_CIFAR10.wide_resnet_layer_input import Wide_ResNet as NetWork
    # from models_CIFAR10lite.resnet import resnet20_cifar as NetWork

    net = NetWork()
    # net = NetWork('VGG-QD')
    # net = NetWork(**{'widen_factor':20, 'depth':28, 'dropout_rate':0.3, 'num_classes':10})
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    # print (net.state_dict().keys())
    model_name = 'AlexNet'
    layer_name_collection = generate_layer_name_collections(net,
                                                            model_name=model_name,
                                                            quantized_first_last_layer=True)
    print (layer_name_collection)
    # conv_name = layer_name_collection[5][0]
    # conv_name = 'module.layer3.2.conv1.weight'
    # conv_name = 'module.fc.weight'
    conv_name = 'module.classifier.1.weight'
    # print ('Process layer: %s' %conv_name)
    _, trainable_names = generate_trainable_parameters(net.named_parameters(),
                                                        conv_name,
                                                        model_name,
                                                        quantized_first_last_layer=True)
    print('------------------------')
    for name in trainable_names:
        print (name)

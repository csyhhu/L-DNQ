import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from datetime import datetime

import tensorflow as tf
import os
import numpy as np


# Construct hessian computing graph for res layer (conv layer without bias)
def create_res_hessian_computing_tf_graph(input_shape, layer_kernel, layer_stride):

    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    patches = tf.extract_image_patches(images = input_holder,
                                       ksizes = [1,layer_kernel, layer_kernel,1],
                                       strides = [1, layer_stride, layer_stride, 1],
                                       rates = [1, 1, 1, 1],
                                       padding = 'SAME')
    print ('Patches shape: %s' %patches.get_shape())
    a = tf.expand_dims(patches, axis=-1)
    b = tf.expand_dims(patches, axis=3)
    outprod = tf.multiply(a, b)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    print ('Hessian shape: %s' % get_hessian_op.get_shape())
    return input_holder, get_hessian_op


# Construct hessian computing graph for fc layer
def create_fc_hessian_computing_tf_graph(input_shape):

    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    a = tf.expand_dims(input_holder, axis=-1)
    vect_w_b = tf.concat([a, tf.ones([tf.shape(a)[0], 1, 1])], axis=1)
    outprod = tf.matmul(vect_w_b, vect_w_b, transpose_b=True)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=0)
    print ('Hessian shape: %s' % get_hessian_op.get_shape())
    return input_holder, get_hessian_op


# Construct hessian computing graph for fc- layer
def create_fc_minus_hessian_computing_tf_graph(input_shape):

    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    a = tf.expand_dims(input_holder, axis=-1)
    outprod = tf.matmul(a, a, transpose_b=True)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=0)
    print ('Hessian shape: %s' % get_hessian_op.get_shape())
    return input_holder, get_hessian_op


# Construct hessian computing graph
def create_conv_hessian_computing_tf_graph(input_shape, layer_kernel, layer_stride):

    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    patches = tf.extract_image_patches(images = input_holder,
                                       ksizes = [1,layer_kernel, layer_kernel,1],
                                       strides = [1, layer_stride, layer_stride, 1],
                                       rates = [1, 1, 1, 1],
                                       padding = 'SAME')
    print ('Patches shape: %s' %patches.get_shape())
    vect_w_b = tf.concat([patches, tf.ones([tf.shape(patches)[0], \
                tf.shape(patches)[1], tf.shape(patches)[2], 1])], axis=3)
    a = tf.expand_dims(vect_w_b, axis=-1)
    b = tf.expand_dims(vect_w_b, axis=3)
    outprod = tf.multiply(a, b)
    # print 'outprod shape: %s' %outprod.get_shape()
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    print ('Hessian shape: %s' % get_hessian_op.get_shape())
    return input_holder, get_hessian_op


def generate_hessian(net, trainloader, layer_name, layer_type, n_batch_used = 100, batch_size = 2, stride_factor = 3):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    if 'module' in layer_name:
        layer_name = layer_name[7:]

    net.eval()
    for batch_idx, (inputs, _) in enumerate(trainloader):

        inputs = inputs.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
        # net(Variable(inputs.cuda(), volatile=True))
        net(inputs)
        layer_input = net.module.layer_input[layer_name]

        # In the begining, construct hessian graph
        if batch_idx == 0:
            print ('[%s] Now construct generate hessian op for layer %s' %(datetime.now(), layer_name))
            # res layer
            if layer_type == 'R':
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_res_hessian_computing_tf_graph(layer_input_np.shape, 
                                                        net.module.layer_kernel[layer_name],
                                                        net.module.layer_stride[layer_name] * stride_factor)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print ('Hessian shape: %d' %hessian_shape)
                weight_shape = net.module.state_dict()['%s.weight' %layer_name].size()
                # print ('Kernel shape: %s' %weight_shape)
                # print weight_shape
                kernel_unfold_shape = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
                print ('Kernel unfold shape: %d' %kernel_unfold_shape)
                assert(hessian_shape == kernel_unfold_shape)
            # linear layer
            elif layer_type == 'F':
                layer_input_np = layer_input.cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_fc_hessian_computing_tf_graph(layer_input_np.shape)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print ('Hessian shape: %d' % hessian_shape)
                weight_shape = net.module.state_dict()['%s.weight' % layer_name].size()
                print ('Weights shape: %d' % weight_shape[1])
                assert(hessian_shape == weight_shape[1] + 1) # +1 because of bias 
            elif layer_type == 'C':
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_conv_hessian_computing_tf_graph(layer_input_np.shape, 
                                                        net.module.layer_kernel[layer_name],
                                                        net.module.layer_stride[layer_name] * stride_factor)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print ('Hessian shape: %d' %hessian_shape)
                weight_shape = net.module.state_dict()['%s.weight' %layer_name].size()
                # print ('Kernel shape: %s' %weight_shape)
                # print weight_shape
                kernel_unfold_shape = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
                print ('Kernel unfold shape: %d' %kernel_unfold_shape)
                assert(hessian_shape == kernel_unfold_shape + 1)
            elif layer_type == 'F-':
                layer_input_np = layer_input.cpu().numpy()
                layer_input_holder, generate_hessian_op = \
                    create_fc_minus_hessian_computing_tf_graph(layer_input_np.shape)
                # check whether dimension is right
                hessian_shape = int(generate_hessian_op.get_shape()[0])
                print ('Hessian shape: %d' % hessian_shape)
                weight_shape = net.module.state_dict()['%s.weight' % layer_name].size()
                print ('Weights shape: %d' % weight_shape[1])
                assert (hessian_shape == weight_shape[1])  # because of no bias

            print ('[%s] %s Graph build complete.'  % (datetime.now(), layer_name))
    
        # Initialization finish, begin to calculate
        if layer_type == 'C' or layer_type == 'R':
            this_layer_input = layer_input.permute(0, 2, 3, 1).cpu().numpy()
        elif layer_type == 'F' or layer_type == 'F-':
            this_layer_input = layer_input.cpu().numpy()

        this_hessian = sess.run(generate_hessian_op,
                                feed_dict={layer_input_holder: this_layer_input})

        if batch_idx == 0:
            layer_hessian = this_hessian
        else:
            layer_hessian += this_hessian

        if batch_idx % 10 == 0:
            print ('[%s] Now finish image No. %d / %d' \
                %(datetime.now(), batch_idx * batch_size, n_batch_used * batch_size))
    
        if batch_idx == n_batch_used:
            break

    net.train()

    return (1.0 / n_batch_used) * layer_hessian


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from models_ImageNet.resnet_layer_input import resnet18 as NetWork
    from utils.dataset import get_dataloader

    use_cuda = torch.cuda.is_available()

    hessian_loader = get_dataloader("ImageNet", 'val', batch_size=2, length=10000)
    print('Length of hessian loader: %d' % (len(hessian_loader)))

    ################
    # Load Models ##
    ################

    net = NetWork()
    # net.load_state_dict(torch.load(pretrain_path))
    if use_cuda:
        print('Dispatch model in %d GPUs' % (torch.cuda.device_count()))
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    layer_collection_list = [('module.fc', '')]

    for layer_idx, layer_collection in enumerate(layer_collection_list):
        conv_name = layer_collection[0]
        bn_name = layer_collection[1]

        print('Now Process to %s' % conv_name)

        if 'fc' in conv_name or 'linear' in conv_name:
            hessian = generate_hessian(net, hessian_loader, conv_name, layer_type='F-')
        else:
            hessian = generate_hessian(net, hessian_loader, conv_name, layer_type='R')

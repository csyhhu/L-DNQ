"""
An utils function for quantization methods
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim

import collections
from datetime import datetime

import numpy as np

# import tensorflow as tf

import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def get_error(theta_B, hessian, theta_0):
    """
    Calculate \delta \theta^T H \delta \theta
    :param theta_B:
    :param hessian:
    :param theta_0:
    :param alpha:
    :param sigma:
    :return:
    """

    delta = theta_B - theta_0
    error = np.trace(np.dot(np.dot(delta.T, hessian), delta))

    return error


def unfold_kernel(kernel):
    """
    In pytorch format, kernel is stored as [out_channel, in_channel, height, width]
    Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
    :param kernel: numpy ndarray
    :return:
    """
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

    return weight


def fold_weights(weights, kernel_shape):
    """
    In pytorch format, kernel is stored as [out_channel, in_channel, width, height]
    Fold weights into a 4-dimensional tensor as [out_channel, in_channel, width, height]
    :param weights:
    :param kernel_shape:
    :return:
    """
    kernel = np.zeros(shape=kernel_shape)
    for i in range(kernel_shape[0]):
        kernel[i,:,:,:] = weights[:, i].reshape([kernel_shape[1], kernel_shape[2], kernel_shape[3]])

    return kernel


def quantize(V, alpha, beta=0, kbits=3):
    """
    Given a real value matrix V, quantize it into {1, 0 ,-1}, according to elements' distance to alpha, beta
    :param V: real value weights matrix
    :return Q: {1, 0 ,-1} matrix
    """
    if kbits == 3:

        if isinstance(V, np.ndarray):
            pos = 1 * (V > (alpha / 2))
            neg = -1 * (V < -(alpha / 2))
        else:
            data_type = type(V)
            pos = (V > (alpha / 2)).type(data_type)
            neg = -1 * (V < -(alpha / 2)).type(data_type)

        Q = pos + neg

    elif kbits == 5:

        if isinstance(V, np.ndarray):
            pos_one = 1 * ((V > (alpha / 2)) & (V < (3 * alpha / 2)))
            pos_two = 2 * (V > (3 * alpha / 2))
            neg_one = -1 * ((V < -(alpha / 2)) & (V > -(3 * alpha / 2)))
            neg_two = -2 * (V < -(3 * alpha / 2))
        else:
            data_type = type(V)
            pos_one = 1 * ((V > (alpha / 2)) & (V < (3 * alpha / 2))).type(data_type)
            pos_two = 2 * (V > (3 * alpha / 2)).type(data_type)
            neg_one = -1 * ((V < -(alpha / 2)) & (V > -(3 * alpha / 2))).type(data_type)
            neg_two = -2 * (V < -(3 * alpha / 2)).type(data_type)

        Q = pos_one + pos_two + neg_one + neg_two

    elif kbits == 7:

        if isinstance(V, np.ndarray):
            pos_one = 1 * ((V > (alpha / 2)) & (V < (3 * alpha / 2)))
            pos_two = 2 * ((V > (3 * alpha / 2)) & (V < (3 * alpha)))
            pos_four = 4 * (V > (3 * alpha))
            neg_one = -1 * ((V < -(alpha / 2)) & (V > -(3 * alpha / 2)))
            neg_two = -2 * ((V < -(3 * alpha / 2)) & (V > -(3 * alpha)))
            neg_four = -4 * (V < -(3 * alpha))
        else:
            data_type = type(V)
            pos_one = 1 * ((V > (alpha / 2)) & (V < (3 * alpha / 2))).type(data_type)
            pos_two = 2 * ((V > (3 * alpha / 2)) & (V < (3 * alpha))).type(data_type)
            pos_four = 4 * (V > (3 * alpha)).type(data_type)
            neg_one = -1 * ((V < -(alpha / 2)) & (V > -(3 * alpha / 2))).type(data_type)
            neg_two = -2 * ((V < -(3 * alpha / 2)) & (V > -(3 * alpha))).type(data_type)
            neg_four = -4 * (V < -(3 * alpha)).type(data_type)

        Q = pos_one + pos_two + neg_one + neg_two + pos_four + neg_four

    elif kbits == 9:

        if isinstance(V, np.ndarray):
            pos_one = 1 * ((V > 0.5 * alpha) & (V < 1.5 * alpha))
            pos_two = 2 * ((V > 1.5 * alpha) & (V < 3 * alpha))
            pos_four = 4 * ((V > 3 * alpha) & (V < 6 * alpha))
            pos_eight = 8 * (V > 6 * alpha)

            neg_one = -1 * ((V < -0.5 * alpha) & (V > -1.5 * alpha))
            neg_two = -2 * ((V < -1.5 * alpha) & (V > -3 * alpha))
            neg_four = -4 * ((V < -3 * alpha) & (V > -6 * alpha))
            neg_eight = -8 * (V < -6 * alpha)
        else:
            data_type = type(V)
            pos_one = 1 * ((V > 0.5 * alpha) & (V < 1.5 * alpha)).type(data_type)
            pos_two = 2 * ((V > 1.5 * alpha) & (V < 3 * alpha)).type(data_type)
            pos_four = 4 * ((V > 3 * alpha) & (V < 6 * alpha)).type(data_type)
            pos_eight = 8 * (V > 6 * alpha).type(data_type)

            neg_one = -1 * ((V < -0.5 * alpha) & (V > -1.5 * alpha)).type(data_type)
            neg_two = -2 * ((V < -1.5 * alpha) & (V > -3 * alpha)).type(data_type)
            neg_four = -4 * ((V < -3 * alpha) & (V > -6 * alpha)).type(data_type)
            neg_eight = -8 * (V < -6 * alpha).type(data_type)

        Q = pos_one + pos_two + pos_four + pos_eight + neg_one + neg_two + neg_four + neg_eight

    elif kbits == 11:

        if isinstance(V, np.ndarray):

            pos_one = 1 * ((V > 0.5 * alpha) & (V <= 1.5 * alpha))
            pos_two = 2 * ((V > 1.5 * alpha) & (V <= 3 * alpha))
            pos_four = 4 * ((V > 3 * alpha) & (V <= 6 * alpha))
            pos_eight = 8 * ((V > 6 * alpha) & (V <= 12 * alpha))
            pos_sixteen = 16 * (V > 12 * alpha)

            neg_one = -1 * ((V < -0.5 * alpha) & (V >= - 1.5 * alpha))
            neg_two = -2 * ((V < -1.5 * alpha) & (V >= - 3 * alpha))
            neg_four = -4 * ((V < -3 * alpha) & (V >= - 6 * alpha))
            neg_eight = -8 * ((V < -6 * alpha) & (V >= - 12 * alpha))
            neg_sixteen = - 16 * (V < - 12 * alpha)

        else:
            data_type = type(V)
            pos_one = 1 * ((V > 0.5 * alpha) & (V <= 1.5 * alpha)).type(data_type)
            pos_two = 2 * ((V > 1.5 * alpha) & (V <= 3 * alpha)).type(data_type)
            pos_four = 4 * ((V > 3 * alpha) & (V <= 6 * alpha)).type(data_type)
            pos_eight = 8 * ((V > 6 * alpha) & (V <= 12 * alpha)).type(data_type)
            pos_sixteen = 16 * (V > 12 * alpha).type(data_type)

            neg_one = -1 * ((V < -0.5 * alpha) & (V >= - 1.5 * alpha)).type(data_type)
            neg_two = -2 * ((V < -1.5 * alpha) & (V >= - 3 * alpha)).type(data_type)
            neg_four = -4 * ((V < -3 * alpha) & (V >= - 6 * alpha)).type(data_type)
            neg_eight = -8 * ((V < -6 * alpha) & (V >= - 12 * alpha)).type(data_type)
            neg_sixteen = - 16 * (V < - 12 * alpha).type(data_type)

        Q = pos_one + pos_two + pos_four + pos_eight + pos_sixteen + \
            neg_one + neg_two + neg_four + neg_eight + neg_sixteen

    else:
        print('Such quantization interval is not implemented yet. Please try one of 3,5,7,9,11')
        raise NotImplementedError

    return Q


def mapping(Q, alpha, beta=0, type=3):
    """
    Given a {-1, 0, +1} quantized matrix Q, map it to G according to different types of mapping
    :param Q:
    :param alpha:
    :param beta:
    :return:
    """
    '''
    if type in {'1bit', 'oneshift', 'twoshift', 'pm4', 'pm8', 'pm16'}:
        G = alpha * Q
    elif type == '2bits':
        G = ((alpha - beta) / 2) * Q + (alpha + beta) / 2
    elif type == '2bits_quadric':
        G = ((alpha + beta) / 2) * np.power(Q, 2) + ((alpha - beta) / 2) * Q
    return G
    '''
    G = alpha * Q
    return G


def symmetric_projection(V, weight, hessian, init_alpha, type, layer_name, error_info_file, \
                         proj_ite_times=10):
    last_alpha = init_alpha
    alpha = init_alpha
    proj_best_alpha = init_alpha

    Q = quantize(V, proj_best_alpha, 0, type)
    proj_best_G = mapping(Q, alpha, 0, type)
    # proj_min_error = get_error(proj_best_G, hessian, weight)
    min_G_V_error = np.sum(np.power(V - proj_best_G, 2))
    if error_info_file is not None:
        error_info_file.write('[Before Projection] G,V error: %f, alpha: %f\n' \
                              % (min_G_V_error, proj_best_alpha))

    for proj_ite in range(proj_ite_times):

        Q = quantize(V, alpha, 0, type)  # Project according to current alpha and beta
        alpha = np.dot(V.reshape(-1).T, Q.reshape(-1)) / np.dot(Q.reshape(-1).T, Q.reshape(-1))

        # Get G under new Q and alpha, beta
        G = mapping(Q, alpha, 0, type)

        G_V_error = np.sum(np.power(V - G, 2))
        # cur_error = get_error(G, hessian, weight)

        if error_info_file is not None:
            error_info_file.write('[Proj ite: %d] G,V error: %f,  alpha: %f\n' \
                                  % (proj_ite, G_V_error, alpha))

        print ('[%s] [Layer %s] [Projection: %d] G_V_error error: %f' \
              % (datetime.now(), layer_name, proj_ite, G_V_error))

        # Use G V error to judge best parameters
        if min_G_V_error > G_V_error:
            min_G_V_error = G_V_error
            proj_best_alpha = alpha
            proj_best_G = G
        '''
        if proj_min_error > cur_error:
            proj_min_error = cur_error
            proj_best_alpha = alpha
            proj_best_G = G
        '''

        #  raw_input()
        print ('[%s] [Layer %s] [Projection: %d] alpha: %f' \
              % (datetime.now(), layer_name, proj_ite, alpha))

        # Whether to jump out of projection
        if np.abs(last_alpha - alpha) < 10e-4:
            print ('Absolute value is small enough !')
            break
        else:
            last_alpha = alpha

    return proj_best_G


def ADMM_quantization(layer_name, layer_type, kernel, bias, hessian, kbits, stat_dict=None, \
                      error_info_root=None, save_root=None, rho_factor=1, ADMM_ite_times=100, proj_method='LDNQ'):
    """
    New version of ADMM_quantization has the following benefits:
    1) Input kernel and bias, return exactly the same dimension kernel and bias.
    2) Adopt layer type to classifier different type of layer, no use layer name anymore

    Params:

    layer_type: 'C' for convolution with bias, 'R' for Res layer without bias,
                'F' for fully-connected with bias
    kernel:     4-dimension or 2-dimension weights
    bias:       Can be None, if layer_type=='R', must be None
    """

    if layer_type == 'C':
        assert (len(kernel.shape) == 4)
        assert (bias is not None)
        weight = unfold_kernel(kernel)  # [64 (out_chan), 3 (in_chan),3,3] => [27, 64] => [28, 64]
        kernel_shape = kernel.shape
        weight = np.concatenate([weight, bias.reshape(1, -1)], axis=0)
        print ('Layer type: %s, unfolded kernel shape: %s' % (layer_type, weight.shape))
    elif layer_type == 'R':
        assert (len(kernel.shape) == 4)
        assert (bias is None)
        weight = unfold_kernel(kernel)  # [3,3,3,64] => [27, 64]
        kernel_shape = kernel.shape
    elif layer_type == 'F':
        assert (len(kernel.shape) == 2)
        assert (bias is not None)
        weight = np.concatenate([kernel.transpose(), bias.reshape(1, -1)], axis=0)  # [512, 10] => [513, 10]
    elif layer_type == 'F-':
        assert (len(kernel.shape) == 2)
        assert (bias is None)
        weight = kernel.transpose()  # [10, 512] => [512, 10]
    else:
        assert (len(kernel.shape) == 2)
        assert (bias is None)
        weight = kernel.transpose()  # [10, 512] => [512, 10]

    l1, l2 = weight.shape  # [l1, l2]

    if proj_method == 'kmeans':
        enc = OneHotEncoder()
        kmeans = KMeans(n_clusters=kbits).fit(weight.reshape([-1, 1]))
        label_oht = enc.fit_transform(kmeans.labels_.reshape([-1, 1]))
        G = label_oht.dot(kmeans.cluster_centers_).reshape(weight.shape)
    elif proj_method == 'dorefa':
        G = dorefa_fw(weight, kbits)
    else:
        alpha = np.sum(np.abs(weight)) / float(weight.size)
        G = alpha * quantize(weight, alpha, 0, kbits=kbits)

    dual = np.zeros(shape=weight.shape)
    rho = rho_factor * hessian[0, 0]

    A = hessian + rho * np.eye(l1)  # [l1, l1]
    b_1 = np.dot(hessian, weight)  # [l1, l1] [l1, l2] => [l1, l2]

    ADMM_min_error = get_error(G, hessian, weight)
    print ('[%s] [Layer %s] Weights shape: %s' % (datetime.now(), layer_name, weight.shape))
    print ('[%s] [Layer %s] Before ADMM error, purely clustering: %f' % (datetime.now(), layer_name, ADMM_min_error))

    if error_info_root is not None:
        error_info_file = open('%s/%s.txt' % (error_info_root, layer_name), 'w+')
        error_info_file.write('Before ADMM error: %f\n' % ADMM_min_error)

    ascend_count = 0

    ADMM_best_G = G

    for ADMM_ite in range(ADMM_ite_times):

        # Proximal Step
        print ('[%s] [Layer %s] [ADMM ite: %d] Begin Proximal Step.' % (datetime.now(), layer_name, ADMM_ite))
        b = b_1 + rho * (G - dual)
        try:
            W = np.linalg.solve(A, b)  # A*W = b : [l1, l1] * [l1, l2] = [l1, l2]
        except Exception:
            print (Exception)
            W = np.linalg.lstsq(A, b)[0]
        print ('[%s] [Layer %s] [ADMM ite: %d] Finish Proximal Step.' % (datetime.now(), layer_name, ADMM_ite))

        # Projection Step
        V = W + dual
        proximal_error = get_error(W, hessian, weight)
        print ('[%s] [Layer %s] [ADMM ite: %d] Proximal Error: %f' % \
              (datetime.now(), layer_name, ADMM_ite, proximal_error))
        if error_info_root is not None:
            error_info_file.write('[ADMM ite: %d] Proximal Error: %f\n' % (ADMM_ite, proximal_error))

        if proj_method == 'kmeans':
            enc = OneHotEncoder()
            kmeans = KMeans(n_clusters=kbits).fit(V.reshape([-1, 1]))
            label_oht = enc.fit_transform(kmeans.labels_.reshape([-1, 1]))
            G = label_oht.dot(kmeans.cluster_centers_).reshape(weight.shape)
        elif proj_method == 'dorefa':
            G = dorefa_fw(V, kbits)
        else:
            G = symmetric_projection(V, weight, hessian, alpha, kbits, layer_name,
                                     None if error_info_root is None else error_info_file)

        proj_error = get_error(G, hessian, weight)

        print ('[%s] [Layer %s] [ADMM ite: %d] Projection Finish. Projection error: %f' \
              % (datetime.now(), layer_name, ADMM_ite, proj_error))

        if error_info_root is not None:
            error_info_file.write('[After projection] Projection error: %f\n' % (proj_error))

        # Dual Update
        dual = dual + W - G

        # Judge when to exit the loop
        if proj_error < ADMM_min_error and np.abs(ADMM_min_error - proj_error) > 10e-4:
            ADMM_min_error = proj_error
            ADMM_best_G = G
            ascend_count = 0
            print ('[Best weights changed]\n')
        else:
            ascend_count += 1

        if ascend_count >= 3:
            print ('Ascend for 3 times, quit. min error: %f' % ADMM_min_error)
            if error_info_root is not None:
                error_info_file.write('Best error: %f\n' % (ADMM_min_error))
            break

        print ('***************************************************************')

    if error_info_root is not None:
        error_info_file.write('Best result in ADMM: \n')
        error_info_file.write('Min error: %f\n' % (ADMM_min_error))

    if stat_dict is not None:
        stat_dict[layer_name] = ADMM_min_error
    print ('[%s] Finish ADMM Process, now saving.' % datetime.now())

    if error_info_root is not None:
        error_info_file.close()
    # Save quantized weights and biases
    if save_root is not None and not os.path.isdir(save_root):
        os.makedirs(save_root)

    if layer_type == 'C':
        quantized_kernel = fold_weights(ADMM_best_G[0:-1, :], kernel_shape)  # [28, 64] => [27, 64] => [64,3,3,3]
        quantized_bias = ADMM_best_G[-1, :].flatten()  # [1, 64] => [64, ]
        if save_root is not None:
            np.save('%s/%s.weight' % (save_root, layer_name), quantized_kernel)
            np.save('%s/%s.bias' % (save_root, layer_name), quantized_bias)
        return quantized_kernel, quantized_bias

    elif layer_type == 'R':
        quantized_kernel = fold_weights(ADMM_best_G, kernel_shape)  # [28, 64] => [27, 64]
        if save_root is not None:
            np.save('%s/%s.weight' % (save_root, layer_name), quantized_kernel)
        return quantized_kernel

    elif layer_type == 'F':
        ADMM_best_G = ADMM_best_G.transpose()  # [513, 10] => [10, 513]
        quantized_weight = ADMM_best_G[:, 0:-1]  # [10, 512]
        quantized_bias = ADMM_best_G[:, -1].flatten()  # [10, ]
        if save_root is not None:
            np.save('%s/%s.weight' % (save_root, layer_name), quantized_weight)
            np.save('%s/%s.bias' % (save_root, layer_name), quantized_bias)
        return quantized_weight, quantized_bias

    elif layer_type == 'F-':
        quantized_weight = ADMM_best_G.transpose()  # [512, 10] => [10, 512]
        if save_root is not None:
            np.save('%s/%s.weight' % (save_root, layer_name), quantized_weight)
        return quantized_weight

    print ('\n\n')


def direct_quantize(param, kbits = 3):
    # print (type(param))

    if isinstance(param, np.ndarray):
        alpha = np.mean(np.abs(param))
    #elif isinstance(param, torch.Tensor) or isinstance(param, torch.cuda.FloatTensor):
    else:
        alpha = torch.mean(torch.abs(param))
    '''else:
        print('Type of arguments 0th not supported.')
        os._exit(999)'''

    Q = quantize(param, alpha, beta=0, kbits=kbits)
    # Get G under new Q and alpha, beta
    G = mapping(Q, alpha)
    return G


def dorefa_quantize(param, kbits = 8):
    n = float(2 ** kbits - 1)
    if isinstance(param, np.ndarray):
        return np.round(param * n) / n
    else:
        return torch.round(param * n) / n

def dorefa_fw(param, bitW = 8):
    if isinstance(param, np.ndarray):
        x = np.tanh(param)
        x = x / np.max(np.abs(x)) * 0.5 + 0.5
    else:
        x = torch.tanh(param)
        x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return 2 * dorefa_quantize(x, bitW) - 1
""" 
This code implements supervised LDNQ in a cascade way
""" 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from models_CIFAR10.resnet_layer_input import *
from models_ImageNet.resnet_layer_input import *
from utils.hessian import generate_hessian
from utils.miscellaneous import generate_trainable_parameters, generate_layer_name_collections, folder_init
from utils.dataset import get_dataloader
from utils.train import cascade_soft_update, validate
from utils.quantization import ADMM_quantization

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
use_cuda = torch.cuda.is_available()

import numpy as np
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='L-DNQ')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model_name', default='ResNet20', type=str, help='Name of model to be quantized')
parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='Name of dataset used')
parser.add_argument('--exp_sec', default='', type=str, help='Specification of this experiments, '
															'which is used for name of saved file')
parser.add_argument('--kbits', default=3, type=int, help='Number of quantized bits')
parser.add_argument('--require_first_test', default=False, type=bool, help='Whether to perform test before quantization to '
																		   'check effectivity of pretrained model')
parser.add_argument('--quantized_first_and_last', default=True, type=bool, help='Whether to quantize first and last layer')

args = parser.parse_args()

# ---------------------------- Configuration --------------------------
model_name = args.model_name
dataset_name = args.dataset_name
exp_spec = args.exp_sec
# Initialize some folder for data saving
folder_init(model_name, ['train_record', 'val_record', 'save_models', \
						 'trainable_names/LDNQ%s' %(exp_spec)])
pretrain_path = './%s/%s_pretrain.pth' %(model_name, model_name)
quantized_path = './%s/save_models/LDNQ%s.pth' %(model_name, exp_spec)
hessian_root = './%s/hessian' %model_name
kbits = args.kbits
trainable_names_record_root = './%s/trainable_names/LDNQ%s' %(model_name, exp_spec)
train_record = open('./%s/train_record/LDNQ%s.txt' %(model_name, exp_spec), 'w')
val_record = open('./%s/val_record/LDNQ%s.txt' %(model_name, exp_spec), 'w')
init_lr = 0.001
# --------------------------------------------------------------------
print ('You are going to quantize model %s into %d bits, using dataset %s, with specification name as %s' \
	   %(model_name, kbits, dataset_name, exp_spec))
input('Press any to continue. Ctrl+C to break.')
################
# Load Dataset #
################
train_loader = get_dataloader(dataset_name, 'limited', batch_size = 128, ratio=0.01)
print ('Length of train loader: %d' %(len(train_loader)))
hessian_loader = get_dataloader(dataset_name, 'limited', batch_size = 2)
print ('Length of hessian loader: %d' %(len(hessian_loader)))
test_loader = get_dataloader(dataset_name, 'test', batch_size = 100)

################
# Load Models ##
################
quantized_net = resnet20_cifar()
# quantized_net = resnet18() # For quantization of ResNet18 using ImageNet
pretrain_param = torch.load(pretrain_path)
quantized_net.load_state_dict(pretrain_param)

original_net = resnet20_cifar()
# original_net = resnet18() # For quantization of ResNet18 using ImageNet
original_net.load_state_dict(pretrain_param)

if use_cuda:
	print('Dispatch model in %d GPUs' % (len(range(torch.cuda.device_count()))))
	quantized_net.cuda()
	quantized_net = torch.nn.DataParallel(quantized_net, device_ids=range(torch.cuda.device_count()))

	original_net.cuda()
	original_net = torch.nn.DataParallel(original_net, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

####################
# First Validation #
####################
if args.require_first_test:
	acc = validate(quantized_net, test_loader, dataset_name=dataset_name)
	print('Full-precision accuracy: %.3f' %acc)
	val_record.write('Full-precision accuracy: %.3f\n' %acc)

# Generate layer name list: layers to be quantized
layer_collection_list = generate_layer_name_collections(
	quantized_net, model_name=model_name, quantized_first_last_layer=args.quantized_first_and_last)

###############
# Begin L-DNQ #
###############
for layer_idx, layer_name in enumerate(layer_collection_list):

	print ('[%s] Process layer %s' % (datetime.now(), layer_name))
	if train_record is not None:
		train_record.write('Process layer %s\n' % layer_name)
	if val_record is not None:
		val_record.write('Process layer %s\n' % layer_name)

	state_dict = quantized_net.state_dict()
	if 'linear' in layer_name or 'fc' in layer_name:
		# Generate Hessian
		hessian = generate_hessian(quantized_net, hessian_loader, layer_name, layer_type='F')
		updated_weight = state_dict['%s.weight' % (layer_name)].cpu().numpy() if use_cuda else \
			state_dict['%s.weight' % (layer_name)].numpy()
		updated_bias = state_dict['%s.bias' % (layer_name)].cpu().numpy() if use_cuda else \
			state_dict['%s.bias' % (layer_name)].numpy()
		# Perform Quantization
		quantized_weight, quantized_bias = ADMM_quantization(layer_name=layer_name, layer_type='F',
															 kernel=updated_weight, bias=updated_bias,
															 hessian=hessian, kbits=kbits)
		state_dict['%s.weight' % (layer_name)] = torch.FloatTensor(quantized_weight)
		state_dict['%s.bias' % (layer_name)] = torch.FloatTensor(quantized_bias)
	else:
		# Generate Hessian
		hessian = generate_hessian(quantized_net, hessian_loader, layer_name, layer_type='R', stride_factor = 1)
		updated_kernel = state_dict['%s.weight' % (layer_name)].cpu().numpy() if use_cuda else \
			state_dict['%s.weight' % (layer_name)].numpy()
		# Perform Quantization
		quantized_kernel = ADMM_quantization(layer_name=layer_name, layer_type='R',
											 kernel=updated_kernel, bias=None, hessian=hessian, kbits=kbits)
		# Step 2: Assignment
		# Assign processed layer with quantized weights
		state_dict['%s.weight' % (layer_name)] = torch.FloatTensor(quantized_kernel)

	###########################
	# Cascaded Weights Update #
	###########################
	quantized_net.load_state_dict(state_dict)
	print ('[%s] Finish layer %s' % (datetime.now(), layer_name))
	# Generate the non-quantized / trainable parameters
	trainable_parameters, trainable_names = \
		generate_trainable_parameters(
			quantized_net.named_parameters(), layer_name + '.weight',
			model_name=model_name, quantized_first_last_layer=args.quantized_first_and_last
		)
	print ('Length of trainable parameters: %d' %(len(trainable_names)))
	trainable_names_record = open('%s/%s.txt' % (trainable_names_record_root, layer_name), 'w')
	for name in trainable_names:
		trainable_names_record.write(name + '\n')
	trainable_names_record.close()

	optimizer = optim.SGD(trainable_parameters, lr=init_lr, momentum=0.9, weight_decay=5e-4)
	cascade_soft_update(quantized_net, original_net, train_loader, dataset_name=dataset_name,
						optimizer=optimizer, train_record=train_record)
	# Record test acc
	acc = validate(quantized_net, test_loader, dataset_name=dataset_name, val_record=val_record)

torch.save(quantized_net.module.state_dict() if use_cuda else quantized_net.state_dict(), quantized_path)
train_record.close()
val_record.close()
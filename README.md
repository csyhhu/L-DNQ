<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


# L-DNQ
Codes for AAAI2019 paper: Deep Neural Network Quantization via Layer-Wise Optimization using Limited Training Data

## How to use it
### Specify your Dataset Root
User need to specify their dataset root in `utils/dataset.py`. For example, in line 39 for CIFAR10 and line 70 for ImageNet.

### Prepare Pre-trained Model
L-DNQ is conducted based on pre-trained model. Therefore you need to:
- CIFAR10 Dataset

run `train_base_model.py` to train a pre-trained model for CIFAR10 dataset. It will generate a pre-trained model in folder `ResNet20`:
    
    cd L-DNQ
    python train_base_model.py
    python train_base_model.py --resume --lr=0.01 # If needed

- ImageNet Dataset

Download the pre-trained model from torchvision model zoo to folder `ResNet18-ImageNet`. For example, download resnet18 from `https://download.pytorch.org/models/resnet18-5c106cde.pth` as is instructed in `models_ImageNet/resnet.py`:

    cd L-DNQ
    mkdir ResNet18-ImageNet
    wget https://download.pytorch.org/models/resnet18-5c106cde.pth
    
### During Quantization
Currently only quantization on ResNet model using CIFAR10/ImageNet dataset is available.  To reproduce ResNet20 quantization using CIFAR10:

    python main.py
    
To reproduce ResNet18 quantization using ImageNet:
    
- Uncomment Line 69, 74 (Comment 68, 73) in `main.py` to change network.


    python main.py --model_name=ResNet18-ImageNet --dataset_name=ImageNet 
    
To reproduce other experiments, please change the network structure accordingly in the code.

## More Specifications

### Change portion of dataset used
In our experiment, only 1% of original dataset is used. If you want to change that portion. Change 

`get_dataloader(dataset_name, 'limited', batch_size = 128, ratio=0.01)` in `main.py` to 

`get_dataloader(dataset_name, 'limited', batch_size = 128, ratio=Whatever you want, resample=True)`

Set `resample` to `False` after new selected dataset is generated. 

If user want to use the whole dataset, simply use `get_dataloader(dataset_name, 'train', batch_size = 128)`

### Change the quantized bits

Change the argument `--kbits` to `3,5,7,9,11`. For example, 5 means the quantization bits are: $0, \pm \alpha, \pm 2*\alpha, \pm 4*\alpha$, totally 5 bits.

## Results
###CIFAR10

| Network | bits | Quantized Acc(%) | Original Acc(%) | Acc Improve(%) |
| ------- | ---- | ------------- | ------------ | ----------- |
| ResNet20|  3   |   85.95       | 91.50        |  -5.55      |
| ResNet32|  3   |   88.47       | 92.13        |  -3.66      |
| ResNet56|  3   |   89.17       | 92.66        |  -3.49      |

###ImageNet

| Network | bits | Quantized Acc(%) | Original Acc(%) | Acc Improve(%) |
| ------- | ---- | ------------- | ------------ | ----------- |
| ResNet18|  3   |   45.98/75.73        | 69.76/89.02        |  -16.43/-10.67      |
| ResNet34|  3   |   43.99/73.05       | 73.30/91.42        |  -29.31/-18.37      |
| ResNet50|  3   |   56.49/81.55       | 76.15/92.87        |  -19.66/-11.32      |

## Requirement
PyTorch > 0.4.0

TensorFlow > 1.3.0

## Support
Please use github issues for any problem related to the code. Send email to the authors for general questions related to the paper.
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = ['AlexNetBN', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetBN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # features.0
            nn.BatchNorm2d(64), # 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # features.4
            nn.BatchNorm2d(192), # 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # features.8
            nn.BatchNorm2d(384), # 9
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # features.11
            nn.BatchNorm2d(256), # 12
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # features.14
            nn.BatchNorm2d(256), # 15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.layer_kernel = {
            'features.0': 11, 
            'features.4': 5, 
            'features.8': 3, 
            'features.11': 3, 
            'features.14': 3
        }

        self.layer_stride = {
            'features.0': 4, 
            'features.4': 1, 
            'features.8': 1, 
            'features.11': 1, 
            'features.14': 1
        }

        self.layer_input = dict()

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        for layer_idx, layers in enumerate(self.features):
            if isinstance(layers, torch.nn.modules.conv.Conv2d):
                self.layer_input['features.%d' %layer_idx] = x.data
            x = layers(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        for layer_idx, layers in enumerate(self.classifier):
            if isinstance(layers, torch.nn.modules.linear.Linear):
                self.layer_input['classifier.%d' %layer_idx] = x.data
            x = layers(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetBN(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

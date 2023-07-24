from models.resnet import ResNet18
from models.densenet import densenet121

# db names
CIFAR10 = 'CIFAR'
MNIST = 'MNIST'

# models as dict
MODELS = {
    'RESNET': ResNet18,
    'DENSENET': densenet121
    }

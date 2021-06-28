import torch.nn as nn


def denormalize(images):
    return (images + 1.0) * 0.5


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

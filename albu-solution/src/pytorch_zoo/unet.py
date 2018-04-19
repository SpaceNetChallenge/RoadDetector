import torch.nn as nn
from pytorch_zoo.abstract_model import EncoderDecoder

class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='resnet34')


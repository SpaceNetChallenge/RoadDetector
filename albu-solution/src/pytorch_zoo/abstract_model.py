import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pytorch_zoo import resnet


encoder_params = {
    'resnet34':
        {'filters': [64, 64, 128, 256, 512],
         'init_op': resnet.resnet34,
         'url': resnet.model_urls['resnet34']}
}

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class PlusBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, dec, enc):
        return enc + dec


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming He normal initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url):
        pretrained_dict = model_zoo.load_url(model_url)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict)

def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])

class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        super().__init__()
        self.filters = encoder_params[encoder_name]['filters']
        self.num_channels = num_channels
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(f * 2, f) for f in reversed(self.filters[:-1])])
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(1, len(self.filters))])

        self.last_upsample = UnetDecoderBlock(self.filters[0], self.filters[0] // 2)
        self.final = self.make_final_classifier(self.filters[0] // 2, num_classes)

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op']()
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if num_channels == 3 and encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'])

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.encoder_stages) - 1:
                enc_results.append(x.clone())

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx])

        x = self.last_upsample(x)
        f = self.final(x)

        return f

    def get_decoder(self, layer):
        return UnetDecoderBlock(self.filters[layer], self.filters[max(layer - 1, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

import torch
import torch.nn as nn

from utils import get_bn

BN = nn.BatchNorm2d


class PretrainedVGG(nn.Module):
    def __init__(self, pretrained_path, features, num_classes, dropout_rate=0.5, bn_mom=0.9):
        super(PretrainedVGG, self).__init__()
        global BN
        BN = get_bn(bn_mom)
        self.features = features
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()
            
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 580),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(580, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )
        self.load_weights(pretrained_path)
    
    def cls_parameters(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))
    
    def load_weights(self, pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        for k in list(filter(lambda s: s.startswith('classifier'), state.keys())):
            state.pop(k)
            
        missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
        assert missing_keys == ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
        assert len(unexpected_keys) == 0
    
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BN(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained_path, num_classes, dropout_rate=0.5, bn_mom=0.9):
    vgg16_cfg = [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M'
    ]
    model = PretrainedVGG(
        pretrained_path=pretrained_path, features=make_layers(vgg16_cfg, True),
        num_classes=num_classes, dropout_rate=dropout_rate, bn_mom=bn_mom
    )
    return model


if __name__ == '__main__':
    v = vgg16(r'C:\Users\16333\Desktop\PyCharm\cv_course\proj_3\homework_set_3\torch_code\models\vgg16-6c64b313.pth', num_classes=15)
    import torchsummary
    torchsummary.summary(v, (3, 224, 224,))
    torchsummary.summary(v.classifier, (512 * 7 * 7,))
    print(v.__class__.__name__)

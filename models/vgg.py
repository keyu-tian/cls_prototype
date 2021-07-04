import torch
import torch.nn as nn

from utils import get_bn, filter_params

BN = nn.BatchNorm2d


class PretrainedVGG(nn.Module):
    def __init__(self, pretrained_path, backbone_cfg, num_classes, dropout_rate=0.6, bn_mom=0.9):
        super(PretrainedVGG, self).__init__()
        global BN
        BN = get_bn(bn_mom)
        
        self.features = PretrainedVGG.make_backbone(backbone_cfg)
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        
        ############  head  ############
        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes),
        )
        self.load_weights(pretrained_path)
    
    def head_parameters(self):
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def make_backbone(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, BN(v), nn.ReLU(inplace=True)]
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
        pretrained_path=pretrained_path, backbone_cfg=vgg16_cfg,
        num_classes=num_classes, dropout_rate=dropout_rate, bn_mom=bn_mom
    )
    return model


if __name__ == '__main__':
    v = vgg16(r'C:\Users\16333\Desktop\PyCharm\cv_course\proj_3\homework_set_3\torch_code\models\vgg16-6c64b313-no-fc.pth', num_classes=15)
    import torchsummary
    torchsummary.summary(v, (3, 224, 224,))
    torchsummary.summary(v.classifier, (512 * 5 * 5,))
    print(v.__class__.__name__)

    print(filter_params(v))

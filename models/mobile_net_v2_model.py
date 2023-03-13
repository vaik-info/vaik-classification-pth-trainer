import torch.nn as nn
import timm

class MobileNetV2Model(nn.Module):
    def __init__(self, class_num):
        super(MobileNetV2Model, self).__init__()
        self.mobile_net_v2 = timm.create_model('mobilenetv2_100', pretrained=True)
        self.mobile_net_v2.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=class_num, bias=True)
        )

    def forward(self, x):
        x = self.mobile_net_v2(x)
        return x
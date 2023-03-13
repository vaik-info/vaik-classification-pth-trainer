import torch.nn as nn
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class MobileNetV2Model(nn.Module):
    def __init__(self, class_num, input_size=256):
        super(MobileNetV2Model, self).__init__()
        self.input_size = input_size
        self.mobile_net_v2 = timm.create_model('mobilenetv2_100', pretrained=True)
        self.mobile_net_v2.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=class_num, bias=True)
        )
        config = resolve_data_config({}, model=self.mobile_net_v2)
        transform = create_transform(**config)
        self.transform = [T.Resize(size=self.input_size),
                                T.CenterCrop(size=self.input_size),
                                transform.transforms[2],
                                transform.transforms[3]]

    def forward(self, x):
        x = self.mobile_net_v2(x)
        return x

    def get_transform(self):
        return self.transform

    def get_normalize_info(self):
        mean = self.transform[-1].mean.to('cpu').detach().numpy().tolist()
        std = self.transform[-1].std.to('cpu').detach().numpy().tolist()
        return mean, std


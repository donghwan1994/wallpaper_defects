import torch
import torch.nn as nn
import torchvision.models as models
import timm.models as timm_models

from dataset import class_name


class BaseModel(nn.Module):
    def __init__(self, num_classes=len(class_name)):
        super(BaseModel, self).__init__()
        self.model = timm_models.deit3_base_patch16_224_in21ft1k(pretrained=True, num_classes=num_classes)
        # self.model = timm_models.vit_base_patch32_224_clip_laion2b(pretrained=True, num_classes=num_classes)
        # self.model = timm_models.efficientnet.tf_efficientnetv2_s_in21ft1k(pretrained=True, num_classes=num_classes)
        # self.model = timm_models.resnet.resnet18(pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    m = BaseModel()
    o = m(x)
    print(o.size())

import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import torch.nn.functional as F
'''model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}'''
class STE_Relu(torch.autograd.Function):
    def forward(self,x):
        return x*(x>0).float()
    def backward(self,g):
        return g

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # feature extractor
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat_0 = [512, 512, 512, 256]

        self.frontend = make_layers(self.frontend_feat,in_channels=3,batch_norm=False,dilation=False)
        self.backend_0 = make_layers(self.backend_feat_0,in_channels = 512,batch_norm=False,dilation = True)

        #density regressor
        self.backend_feat_1 = [128, 64]
        self.backend_1 = make_layers(self.backend_feat_1, in_channels=256, batch_norm=False, dilation=True)
        self.output_layer_1 = nn.Conv2d(64, 1, kernel_size=1)

        # 3 surrogate tasks
        self.backend_feat_2 = [128, 64]
        self.backend_feat_3 = [128, 64]
        self.backend_feat_4 = [128, 64]
    

        self.backend_2 = make_layers(self.backend_feat_2, in_channels=256, batch_norm=False, dilation=True)
        self.backend_3 = make_layers(self.backend_feat_3, in_channels=256, batch_norm=False, dilation=True)
        self.backend_4 = make_layers(self.backend_feat_4, in_channels=256, batch_norm=False, dilation=True)

        self.output_layer_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.output_layer_3 = nn.Conv2d(64, 2, kernel_size=1)
        self.output_layer_4 = nn.Conv2d(64, 2, kernel_size=1)

        # init pre-trained model
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend_0(x)
       
        u = self.backend_1(x)
        u = self.output_layer_1(u)
      
        u = F.relu(u)

        logits2 = self.backend_2(x)
        logits2 = self.output_layer_2(logits2)
        
        logits3 = self.backend_3(x)
        logits3 = self.output_layer_3(logits3)
        
        logits4 = self.backend_4(x)
        logits4 = self.output_layer_4(logits4)
       
        #u is the predicted density map,logits is the prediction of surrogate tasks
        return u,logits2,logits3,logits4 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v=='U':
            layers +=[nn.UpsamplingBilinear2d(scale_factor=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                

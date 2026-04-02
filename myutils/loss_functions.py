import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:-6]
       # checkpoint = torch.load('/run/media/mlcv/DATA_SSD/FaceFrontalization/VGGFace/best_checkpoint.pth',map_location='cpu')
      #  self.vgg.load_state_dict(checkpoint['model'])
        for param in self.vgg.parameters():
                param.requires_grad = False
        vgg_pretrained_features = self.vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        Xfeats = self.vgg(X)

        return out, Xfeats


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()   

        self.device = device 
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, predictions, y):   
        
                
        self.vgg.eval()           
        x_vgg, x_feats = self.vgg(predictions)
        y_vgg, y_feats = self.vgg(y)

        lossvgg = 0
        for i in range(len(x_vgg)):
           lossvgg += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())      
        
        return lossvgg
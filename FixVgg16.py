import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixVgg16(nn.Module):
    def __init__(self):
        super(FixVgg16,self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        for p in self.parameters():
            p.requires_grad = False

        #[64,224,224]
        self.fix1 = nn.Sequential(*list(self.vgg16.children())[0][0:3])
        #[128,112,112]
        self.fix2 = nn.Sequential(*list(self.vgg16.children())[0][0:8])
        #[256,56,56]
        self.fix3 = nn.Sequential(*list(self.vgg16.children())[0][0:15])
        #[512,28,28]
        self.fix4 = nn.Sequential(*list(self.vgg16.children())[0][0:22])
        #[512,14,14]
        self.fix5 = nn.Sequential(*list(self.vgg16.children())[0][0:26])
        #[512,7,7]
        # self.fix6 = nn.Sequential(*list(self.vgg16.children())[0][0:29])

        self.relu = nn.ReLU()


    def hook(self,module,input,output):
        out = F.adaptive_avg_pool2d(output,(1,1))
        self.feature_box.append(out)

    def forward(self,v):

        v1 = F.adaptive_avg_pool2d(self.relu(self.fix1(v)),(1,1))
        v2 = F.adaptive_avg_pool2d(self.relu(self.fix2(v)),(1,1))
        v3 = F.adaptive_avg_pool2d(self.relu(self.fix3(v)),(1,1))
        v4 = F.adaptive_avg_pool2d(self.relu(self.fix4(v)),(1,1))
        v5 = F.adaptive_avg_pool2d(self.relu(self.fix5(v)),(1,1))
        # v6 = F.adaptive_avg_pool2d(self.relu(self.fix6(v)),(1,1))

        vv = torch.cat((v1,v2,v3,v4,v5),1).permute(0,3,2,1)

        return vv

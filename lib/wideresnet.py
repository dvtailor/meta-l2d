import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from lib.attention import MultiHeadAttn, SelfAttn
from lib.datasets import UnNormalize

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNetBase(nn.Module):
    # WRN without final dense layer ("feature extractor")
    def __init__(self, depth, n_channels, widen_factor=1, dropRate=0.0):
        super(WideResNetBase, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(n_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out
    

class Classifier(nn.Module):
    def __init__(self, base_model, num_classes, n_features):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(n_features, num_classes+1)
        self.fc.bias.data.zero_()
        self.params = nn.ModuleDict({
            'base': nn.ModuleList([self.base_model]),
            'clf' : nn.ModuleList([self.fc])
        })

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc(out)
        return out


def build_mlp(dim_in, dim_hid, dim_out, depth):
    if depth==1:
        modules = [nn.Linear(dim_in, dim_out)] # no hidden layers
    else: # depth>1
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class ClassifierRejectorWithContextEmbedder(nn.Module):
    # Instantiate with actual num_classes (not augmented)
    def __init__(self, base_model, num_classes, n_features, dim_hid=128, depth_embed=6, depth_rej=4, with_attn=False):
        super(ClassifierRejectorWithContextEmbedder, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()
        self.rejector = build_mlp(n_features+dim_hid, dim_hid, 1, depth_rej)
        self.embed_class = nn.Embedding(num_classes, dim_hid)

        # NB: just for testing purposes (shouldn't hard-code dataset scaling in ClassifierRejector)
        unnormalize_cifar10 = UnNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        weights_pretrained = ResNet18_Weights.IMAGENET1K_V1
        self.transform_imagenet = transforms.Compose([unnormalize_cifar10,weights_pretrained.transforms()])
        self.model_pretrained = resnet18(weights=weights_pretrained)
        for param in self.model_pretrained.parameters():
            param.requires_grad = False
        dim_resnet18_out = self.model_pretrained.fc.in_features
        self.model_pretrained.fc = Identity() # override final layer to instead output penultimate activations (dim=512)
        self.model_pretrained.eval()
        self.affine = nn.Linear(dim_resnet18_out, dim_hid, bias=False) # project down pretrained feature extractor

        rej_mdl_lst = [self.rejector, self.embed_class, self.affine]
        if not with_attn:
            self.embed = build_mlp(3*dim_hid, dim_hid, dim_hid, depth_embed)
            rej_mdl_lst += [self.embed]
        else:
            self.embed = nn.Sequential(
                build_mlp(3*dim_hid, dim_hid, dim_hid, depth_embed),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.embed, self.attn]
        
        self.params = nn.ModuleDict({
            'base': nn.ModuleList([self.base_model]),
            'clf' : nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })

    def forward(self, x, cntxt):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        n_experts = cntxt.xc.shape[0]
        
        x_embed = self.base_model(x) # [B,Dx]
        logits_clf = self.fc(x_embed) # [B,K]
        logits_clf = logits_clf.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K]

        embedding = self.encode(cntxt, x) # [E,B,H]
        x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
        packed = torch.cat([x_embed,embedding], -1) # [B,Dx+H] -> [E,B,Dx+H]
        logit_rej = self.rejector(packed) # [E,B,1]
        
        out = torch.cat([logits_clf,logit_rej], -1) # [E,B,K+1]
        return out
    
    def encode(self, cntxt, xt):
        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:]) # [E*Nc,3,32,32]
        xc_embed = self.model_pretrained(self.transform_imagenet(cntxt_xc)).data # [E*Nc,512]
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,512]
        xc_embed = self.affine(xc_embed) # [E,Nc,H]

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]

        out = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,3H]
        out = self.embed(out) # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2) # [E,H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size,1) # [E,B,H]
        else:
            xt_embed = self.model_pretrained(self.transform_imagenet(xt)).data # [B,512]
            xt_embed = self.affine(xt_embed) # [B,H]
            xt_embed = xt_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,H]
            embedding = self.attn(xt_embed, xc_embed, out) # [E,B,H]
        
        return embedding

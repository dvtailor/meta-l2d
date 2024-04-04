import copy
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention import MultiHeadAttn, SelfAttn

class Classifier(nn.Module):
    def __init__(self, base_model, num_classes, n_features, with_softmax=True):
        super(Classifier, self).__init__()
        self.base_model = base_model
        
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.with_softmax = with_softmax

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc(out) # [B,K]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
    

class ClassifierRejector(nn.Module):
    def __init__(self, base_model, num_classes, n_features, with_softmax=True, decouple=False):
        super(ClassifierRejector, self).__init__()
        self.base_model_clf = base_model
        base_mdl_lst = [self.base_model_clf]
        if decouple:
            self.base_model_rej = copy.deepcopy(self.base_model_clf)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model_clf

        self.fc_clf = nn.Linear(n_features, num_classes)
        self.fc_clf.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, 1)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc_clf,self.fc_rej])
        })

    def forward(self, x):
        out = self.base_model_clf(x)
        logits_clf = self.fc_clf(out) # [B,K]

        out = self.base_model_rej(x)
        logit_rej = self.fc_rej(out) # [B,1]

        out = torch.cat([logits_clf,logit_rej], -1) # [B,K+1]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out


def get_activation(act_str):
    if act_str == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif act_str == 'elu':
        return functools.partial(nn.ELU, inplace=True)
    else:
        raise ValueError('invalid activation')


def build_mlp(dim_in, dim_hid, dim_out, depth, activation='relu'):
    act = get_activation(activation)
    if depth==1:
        modules = [nn.Linear(dim_in, dim_out)] # no hidden layers
    else: # depth>1
        modules = [nn.Linear(dim_in, dim_hid), act()]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(act())
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class ClassifierRejectorWithContextEmbedder(nn.Module):
    def __init__(self, base_model, num_classes, n_features, dim_hid=128, depth_embed=6, depth_rej=4, dim_class_embed=128,
                 with_attn=False, with_softmax=True, decouple=False):
        super(ClassifierRejectorWithContextEmbedder, self).__init__()
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.with_softmax = with_softmax
        self.decouple = decouple
        self.dim_hid = dim_hid
        self.n_features = n_features

        self.base_model = base_model
        base_mdl_lst = [self.base_model]
        if self.decouple:
            self.base_model_rej = copy.deepcopy(self.base_model)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model
        
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.embed_class = nn.Embedding(num_classes, dim_class_embed)    
        self.rejector = build_mlp(n_features+dim_hid, dim_hid, 1, depth_rej)
        self.rejector[-1].bias.data.zero_()

        if not self.with_attn:
            self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)
        else:
            self.embed = nn.Sequential(
                build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed-2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )
        
        rej_mdl_lst = [self.rejector, self.embed_class, self.embed]

        if with_attn:
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.attn]
        
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })

    def forward(self, x, cntxt=None):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        if cntxt is None:
            n_experts = 1
        else:
            n_experts = cntxt.xc.shape[0]
        
        x_embed = self.base_model(x) # [B,Dx]
        logits_clf = self.fc(x_embed) # [B,K]
        logits_clf = logits_clf.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K]

        if cntxt is None:
            embedding = torch.zeros((n_experts, x.shape[0], self.dim_hid), device=x_embed.device)
        else:
            embedding = self.encode(cntxt, x) # [E,B,H]
        
        x_embed = self.base_model_rej(x) # [B,Dx]
        x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]

        packed = torch.cat([x_embed,embedding], -1) # [E,B,Dx+H]
        
        logit_rej = self.rejector(packed) # [E,B,1]
        
        out = torch.cat([logits_clf,logit_rej], -1) # [E,B,K+1]
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
    
    def encode(self, cntxt, xt):
        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:]) # [E*Nc,3,32,32]

        # stop gradient flow to base model
        # for coupled architecture (shared WRN) backprop here could be detrimental to classifier performance; maybe OK for decoupled architecture
        if not self.decouple:
            with torch.no_grad():
                xc_embed = self.base_model_rej(cntxt_xc) # [E*Nc,Dx]
            xc_embed = xc_embed.detach()
        else:
            xc_embed = self.base_model_rej(cntxt_xc) # [E*Nc,Dx]
        
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,Dx]

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]
        out = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]

        out = self.embed(out) # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2) # [E,H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size,1) # [E,B,H]
        else:
            xt_embed = self.base_model_rej(xt) # [B,Dx]
            if not self.decouple:
                xt_embed = xt_embed.detach() # stop gradients flowing
            xt_embed = xt_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
            embedding = self.attn(xt_embed, xc_embed, out) # [E,B,H]
        
        return embedding

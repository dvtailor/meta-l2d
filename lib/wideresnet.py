import copy
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention import MultiHeadAttn, SelfAttn
from lib.lbanp_modules import TransformerEncoderLayer, TransformerEncoder, NPDecoderLayer, NPDecoder

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

        self.base_model = base_model
        base_mdl_lst = [self.base_model]
        if self.decouple:
            self.base_model_rej = copy.deepcopy(self.base_model)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model
        
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.rejector = build_mlp(n_features+dim_hid, dim_hid, 1, depth_rej)
        self.rejector[-1].bias.data.zero_()
        self.embed_class = nn.Embedding(num_classes, dim_class_embed)
        
        if not with_attn:
            self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)
        else:
            self.embed = nn.Sequential(
                build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed-2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )
        # self.embed_post = build_mlp_fixup(dim_hid, dim_hid, dim_hid, 2)
        rej_mdl_lst = [self.rejector, self.embed_class, self.embed] #self.embed_post

        if with_attn:
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.attn]
        
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
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
        xc_embed = self.base_model_rej(cntxt_xc) # [E*Nc,Dx]
        # stop gradient flow to base model
        # for coupled architecture (shared WRN) backprop here could be detrimental to classifier performance; maybe OK for decoupled architecture
        if not self.decouple:
            xc_embed = xc_embed.detach()
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,Dx]

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]
        out = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]

        out = self.embed(out) # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2) # [E,H]
            # embedding = self.embed_post(embedding) # [E,H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size,1) # [E,B,H]
        else:
            xt_embed = self.base_model_rej(xt) # [B,Dx]
            if not self.decouple:
                xt_embed = xt_embed.detach() # stop gradients flowing
            xt_embed = xt_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
            embedding = self.attn(xt_embed, xc_embed, out) # [E,B,H]
        
        return embedding


# # use reduced "latent" dimensionality of 64
# # but introduce new arg "dim_feedforward" that is used for rejector width and transformer architecture
# # MLP embedder uses dim_hid
# # rejector depth shallower (2) compared to before (same as TNP)
# class ClassifierRejectorWithContextEmbedderTransformer(nn.Module):
#     def __init__(self, base_model, num_classes, n_features, dim_hid=64, depth_embed=4, depth_rej=2, dim_class_embed=128,
#                  with_softmax=True, decouple=False, dim_feedforward=128):
#         super(ClassifierRejectorWithContextEmbedderTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.with_softmax = with_softmax
#         self.decouple = decouple

#         self.base_model = base_model
#         base_mdl_lst = [self.base_model]
#         if self.decouple:
#             self.base_model_rej = copy.deepcopy(self.base_model)
#             base_mdl_lst += [self.base_model_rej]
#         else:
#             self.base_model_rej = self.base_model
        
#         self.fc = nn.Linear(n_features, num_classes)
#         self.fc.bias.data.zero_()

#         # TNP drops xt from input, i.e. build_mlp(dim_hid,...)
#         # NOTE: preferable to revert to rejector explicitly as function of xt since this is how written in paper
#         # self.rejector = build_mlp(n_features+dim_hid, dim_feedforward, 1, depth_rej)
#         self.rejector = build_mlp(dim_hid, dim_feedforward, 1, depth_rej)
#         self.rejector[-1].bias.data.zero_()

#         self.embed_class = nn.Embedding(num_classes, dim_class_embed)
#         self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)

#         rej_mdl_lst = [self.rejector, self.embed_class, self.embed]

#         layers_transformer = 6
#         encoder_layer = TransformerEncoderLayer(dim_hid, nhead=4, dim_feedforward=dim_feedforward, dropout=0)
#         self.encoder = TransformerEncoder(encoder_layer, layers_transformer-1)
#         rej_mdl_lst += [self.encoder]

#         self.embed_query = build_mlp(n_features, dim_hid, dim_hid, depth_embed)
#         decoder_layer = NPDecoderLayer(dim_hid, nhead=4, dim_feedforward=dim_feedforward, dropout=0, norm_first=True)
#         self.decoder = NPDecoder(decoder_layer, layers_transformer)
#         self.norm = nn.LayerNorm(dim_hid)
#         rej_mdl_lst += [self.embed_query, self.decoder, self.norm]
        
#         self.params = nn.ModuleDict({
#             'base': nn.ModuleList(base_mdl_lst),
#             'clf' : nn.ModuleList([self.fc]),
#             'rej': nn.ModuleList(rej_mdl_lst)
#         })

#     def forward(self, x, cntxt):
#         '''
#         Args:
#             x : tensor [B,3,32,32]
#             cntxt : AttrDict, with entries
#                 xc : tensor [E,Nc,3,32,32]
#                 yc : tensor [E,Nc]
#                 mc : tensor [E,Nc]
#         '''
#         n_experts = cntxt.xc.shape[0]
        
#         x_embed = self.base_model(x) # [B,Dx]
#         logits_clf = self.fc(x_embed) # [B,K]
#         logits_clf = logits_clf.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K]

#         embedding = self.encode(cntxt, x) # [E,B,H]
#         # x_embed = self.base_model_rej(x) # [B,Dx]
#         # x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
#         # packed = torch.cat([x_embed,embedding], -1) # [E,B,Dx+H]
#         packed = embedding # NEW
#         logit_rej = self.rejector(packed) # [E,B,1]
        
#         out = torch.cat([logits_clf,logit_rej], -1) # [E,B,K+1]
#         if self.with_softmax:
#             out = F.softmax(out, dim=-1)
#         return out
    
#     def encode(self, cntxt, xt):
#         n_experts = cntxt.xc.shape[0]
#         batch_size = xt.shape[0]

#         cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:]) # [E*Nc,3,32,32]
#         xc_embed = self.base_model_rej(cntxt_xc) # [E*Nc,Dx]
#         if not self.decouple:
#             xc_embed = xc_embed.detach()
#         xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,Dx]

#         yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
#         mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]
#         embedding_cntx = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]

#         embedding_cntx = self.embed(embedding_cntx) # [E,Nc,H]
#         embedding_cntx = self.encoder(embedding_cntx) # list of tensors with shp [E,Nc,H]
        
#         xt_embed = self.base_model_rej(xt) # [B,Dx]
#         if not self.decouple:
#             xt_embed = xt_embed.detach() # stop gradients flowing
#         # xt_embed = xt_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
#         # embedding_query = self.embed_query(xt_embed) # [E,B,H]
#         embedding_query = self.embed_query(xt_embed) # [B,H]
#         embedding_query = embedding_query.unsqueeze(0).repeat(n_experts,1,1) # [E,B,H]

#         embedding = self.decoder(embedding_query, embedding_cntx) # [E,B,H]
#         embedding = self.norm(embedding) # [E,B,H]
        
#         return embedding

# use reduced "latent" dimensionality of 64
# but introduce new arg "dim_feedforward" that is used for rejector width and transformer architecture
# MLP embedder uses dim_hid
# rejector depth shallower (2) compared to before (same as TNP)
class ClassifierRejectorWithContextEmbedderTransformer(nn.Module):
    def __init__(self, base_model, num_classes, n_features, dim_hid=64, depth_embed=4, depth_rej=2, dim_class_embed=128,
                 with_softmax=True, decouple=False, dim_feedforward=128):
        super(ClassifierRejectorWithContextEmbedderTransformer, self).__init__()
        self.num_classes = num_classes
        self.with_softmax = with_softmax
        self.decouple = decouple
        self.dim_class_embed = dim_class_embed

        self.base_model = base_model
        base_mdl_lst = [self.base_model]
        if self.decouple:
            self.base_model_rej = copy.deepcopy(self.base_model)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model
        
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        # TNP drops xt from input, i.e. build_mlp(dim_hid,...)
        # NOTE: preferable to revert to rejector explicitly as function of xt since this is how written in paper
        # self.rejector = build_mlp(n_features+dim_hid, dim_feedforward, 1, depth_rej)
        self.rejector = build_mlp(dim_hid, dim_feedforward, 1, depth_rej)
        self.rejector[-1].bias.data.zero_()

        self.embed_class = nn.Embedding(num_classes, dim_class_embed)
        self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)

        rej_mdl_lst = [self.rejector, self.embed_class, self.embed]

        layers_transformer = 6
        encoder_layer = nn.TransformerEncoderLayer(dim_hid, nhead=4, dim_feedforward=dim_feedforward, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers_transformer)
        rej_mdl_lst += [self.encoder]
        
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
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
        # x_embed = self.base_model_rej(x) # [B,Dx]
        # x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
        # packed = torch.cat([x_embed,embedding], -1) # [E,B,Dx+H]
        packed = embedding # NEW
        logit_rej = self.rejector(packed) # [E,B,1]
        
        out = torch.cat([logits_clf,logit_rej], -1) # [E,B,K+1]
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
    
    def encode(self, cntxt, xt):
        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:]) # [E*Nc,3,32,32]
        xc_embed = self.base_model_rej(cntxt_xc) # [E*Nc,Dx]
        if not self.decouple:
            xc_embed = xc_embed.detach()
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,Dx]

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]
        embedding_cntx = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]

        xt_embed = self.base_model_rej(xt) # [B,Dx]
        if not self.decouple:
            xt_embed = xt_embed.detach() # stop gradients flowing
        embedding_trgt = torch.cat([
                            xt_embed,
                            torch.zeros((batch_size,self.dim_class_embed)).to(xt_embed.device),
                            torch.zeros((batch_size,self.dim_class_embed)).to(xt_embed.device)], -1) # [B,Dx+2H]
        embedding_trgt = embedding_trgt.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx+2H]
        inp = torch.cat((embedding_cntx, embedding_trgt), dim=1) # [E,Nc+B,Dx+2H]

        # construct mask
        n_cntx = cntxt.xc.shape[1]
        num_all = n_cntx + batch_size
        mask = torch.zeros(num_all, num_all, device=inp.device).fill_(float('-inf'))
        mask[:, :n_cntx] = 0.0

        embeddings = self.embed(inp) # [E,Nc+B,H]
        out = self.encoder(embeddings, mask=mask) # [E,Nc+B,H]
        out = out[:, -batch_size:] # [E,B,H]
        return out

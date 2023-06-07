from model.vision_transformer import ViT
import torch
import torch.nn as nn

# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)


    def forward(self, x):
        features,features2 = self.base(x)
        f1, f2 = features.chunk(2, 0)
        #--------max和avg------
        f3_1,f4_1=features2.chunk(2,0)
        B =f3_1.shape[0]
        f3 = f3_1.permute(0, 2, 1).reshape((B, self.in_planes, 21, 10))
        f4 = f4_1.permute(0, 2, 1).reshape((B, self.in_planes, 21, 10))
        ir=nn.AvgPool2d(kernel_size=(21, 10))
        rgb=nn.MaxPool2d(kernel_size=(21, 10))
        f3=rgb(f3).squeeze()
        f4=ir(f4).squeeze()
        Qrgb=torch.mm(f1, f3.t())
        Qir=torch.mm(f2, f4.t())
        rrr = nn.Softmax(1)(Qrgb)
        iii = nn.Softmax(1)(Qir)
        x_rgb = torch.mm(rrr, f1)
        x_ir = torch.mm(iii, f2)
        #-------交叉—---
        sim_rgbtoir = torch.mm(f1, f2.t())
        sim_irtorgb = torch.mm(f2, f1.t())
        sim_rgbtoir = nn.Softmax(1)(sim_rgbtoir)
        sim_irtorgb = nn.Softmax(1)(sim_irtorgb)
        x_rgbtoir = torch.mm(sim_rgbtoir, f2)
        x_irtorgb = torch.mm(sim_irtorgb, f1)
        f_r=f1+x_rgbtoir+x_rgb
        f_i=f2+x_irtorgb+x_ir
        features1=torch.cat([f_i,f_r])
        feat1 = self.bottleneck(features1)
        feat = self.bottleneck(features)
        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, features,features1

        else:
            return self.l2norm(feat1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from log import Logger
from util import r2, mse, rmse, mae, pp_mse, pp_rmse, pp_mae, pp_r2


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlurPool(nn.Module):
    def __init__(self, stride):
        super(BlurPool, self).__init__()
        self.kernel = nn.Parameter(torch.from_numpy((np.array([[1, 4, 6, 4, 1],
                                              [4, 16, 24, 16, 4],
                                              [6, 24, 36, 24, 6],
                                              [4, 16, 24, 16, 4],
                                              [1, 4, 6, 4, 1]])/256.0).astype('float32')),
                                   requires_grad=False).view(1, 1, 5, 5)
        self.stride = stride

    def forward(self, x):
        num_dims = x.size(1)
        kernel = self.kernel.repeat(num_dims, 1, 1, 1).to(x.device)
        x = F.conv2d(x, kernel, groups=num_dims, stride=self.stride, padding=2)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, anti_alias=False):
        super(BasicBlock, self).__init__()

        if anti_alias and stride != 1:
            self.conv1 = nn.Sequential(conv3x3(inplanes, planes, 1),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True),
                                       BlurPool(stride=stride))
        else:
            self.conv1 = nn.Sequential(conv3x3(inplanes, planes, stride),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True))

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_size, num_dims, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X C X N
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        q_k = math.sqrt(num_dims // 2)
        attention = F.softmax(energy/q_k, dim=2)  # BX (N) X (N)
        proj_value = x.view(batch_size, num_dims, -1)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_dims, width, height)

        out = self.value_conv(out)

        out = self.bn(out)

        out = out + x
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, in_dims=128):
        super(PositionalEncoding, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims+2, in_dims, 1, stride=1, bias=False),
                                  nn.BatchNorm2d(in_dims),
                                  nn.ReLU())

    def forward(self, x):
        batch_size, num_dims, width, height = x.size()
        width_axis = torch.arange(-width//2, width//2, step=1, dtype=x.dtype,
                                  device=x.device).view(1, 1, width, 1).repeat(1, 1, 1, height)
        height_axis = torch.arange(-height//2, height//2, step=1, dtype=x.dtype,
                                   device=x.device).view(1, 1, 1, height).repeat(1, 1, width, 1)
        axis = torch.cat((width_axis, height_axis), dim=1).repeat(batch_size, 1, 1, 1)
        x = torch.cat((x, axis), dim=1)
        x = self.conv(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=128, zero_init_residual=True, non_local=False,
                 anti_alias=False):
        super(ResNet, self).__init__()
        self.anti_alias = anti_alias
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], anti_alias=anti_alias)
        if non_local:
            self.layer1 = nn.Sequential(self.layer1,
                                        SelfAttn(64))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer2 = nn.Sequential(self.layer2,
                                        SelfAttn(128))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer3 = nn.Sequential(self.layer3,
                                        SelfAttn(256))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer4 = nn.Sequential(self.layer4,
                                        SelfAttn(512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, SelfAttn):
                    nn.init.constant_(m.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, anti_alias=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, 1),
                              BlurPool(stride=stride))
                if anti_alias else conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, anti_alias=anti_alias))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.num_parameters = args.num_parameters
        self.nn = nn.Sequential(nn.Conv2d(1+self.num_parameters, 64, 3, stride=2, bias=False),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(64, 128, 3, stride=2, bias=False),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(128, 256, 3, stride=2, bias=False),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(256, 512, 3, stride=2, bias=False),
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(512, 1, 3, stride=1, bias=True))

    def forward(self, x):
        return self.nn(x)


class G(nn.Module):
    def __init__(self, args, emb_nn):
        super(G, self).__init__()
        self.num_parameters = args.num_parameters
        self.nn = emb_nn
        self.mlp = nn.Sequential(nn.Linear(128*8, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(512),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(256),
                                 nn.Linear(256, self.num_parameters))

    def forward(self, x):
        batch_size, num_imgs, width, height = x.size()
        x = x.view(batch_size*num_imgs, 1, width, height)
        x = self.nn(x)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x


class D(nn.Module):
    def __init__(self, args, emb_nn):
        super(D, self).__init__()
        self.num_parameters = args.num_parameters
        self.nn = emb_nn
        self.mlp = nn.Sequential(nn.Linear(128*8+self.num_parameters, 512),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(256, 1))

    def forward(self, x, y):
        batch_size, num_imgs, width, height = x.size()
        x = x.view(batch_size*num_imgs, 1, width, height)
        x = self.nn(x)
        x = x.view(batch_size, -1)
        x = torch.cat((x, y), dim=1)
        x = self.mlp(x)
        return x


class SimpleD(nn.Module):
    def __init__(self, args, emb_nn=None):
        super(SimpleD, self).__init__()
        self.num_parameters = args.num_parameters
        self.nn = emb_nn
        self.mlp = nn.Sequential(nn.Linear(self.num_parameters, 512),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(256, 1))

    def forward(self, x, y):
        return self.mlp(y)


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    This is copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.nn == 'resnet18':
            self.nn = resnet18(non_local=args.non_local, anti_alias=args.anti_alias)
        elif args.nn == 'resnet34':
            self.nn = resnet34(non_local=args.non_local, anti_alias=args.anti_alias)
        elif args.nn == 'resnet50':
            self.nn = resnet50(non_local=args.non_local, anti_alias=args.anti_alias)
        elif args.nn == 'resnet101':
            self.nn = resnet101(non_local=args.non_local, anti_alias=args.anti_alias)
        elif args.nn == 'resnet152':
            self.nn = resnet152(non_local=args.non_local, anti_alias=args.anti_alias)
        self.G = G(args, emb_nn=self.nn)
        if args.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif args.loss == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise Exception("Not implemented")
        if args.optimizer == 'sgd':
            self.opt = optim.SGD(self.G.parameters(), lr=args.lr * 10, momentum=args.momentum, weight_decay=1e-4)
        else:
            self.opt = optim.Adam(self.G.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[75, 150, 200], gamma=0.1)
        self.device = torch.device('cuda')
        self.G = self.G.to(self.device)
#         if torch.cuda.device_count() > 1:
#             self.G = nn.DataParallel(self.G)
        self.model_path = args.model_path
        if self.model_path != '':
            self.load(self.model_path)
        self.exp_path = os.path.join('checkpoints', args.exp_name)
        self.logger = Logger(args=args, path=self.exp_path, metrics=['loss', 'avg_r2', 'mse', 'rmse', 'mae',
                                                                     'pp_mse', 'pp_rmse', 'pp_mae', 'pp_r2',
                                                                     'label', 'pred_label'])
        self.best_test_loss = np.inf
        if args.use_psf:
            psf = np.array(np.loadtxt('PSFevalMatrix.txt', delimiter=','))
        else:
            psf = np.zeros(46)
        self.psf = torch.from_numpy(psf.astype('float32')).view(-1, 46).to(self.device)
        if args.scale_param:
            param_scale = 1 / 0.07 * np.ones(46)
            param_scale[0:6] = [1/16.0, 1/6.0, 1/16.0, 1/15.0, 1/13.0, 1/13.0]
        else:
            param_scale = np.ones(46)
        self.param_scale = torch.from_numpy(param_scale.astype('float32')).view(-1, 46).to(self.device)

    def predict(self, x):
        x = self.G(x)
        return x

    def train_one_epoch(self, loader, epoch):
        self.G.train()
        self.scheduler.step()
        train_loss = 0.0
        num_examples = 0
        pred_labels = []
        labels = []
        for data in tqdm(loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            self.opt.zero_grad()
            pred_label = self.predict(img)
            pred_labels.append(pred_label.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
            loss = self.criterion(pred_label*self.param_scale, label*self.param_scale) \
                   + torch.sum((self.psf*(pred_label-label))**2)
            loss.backward()
            self.opt.step()
            batch_size = img.size(0)
            train_loss += loss.item() * batch_size
            num_examples += batch_size
        pred_label = np.concatenate(pred_labels, axis=0)
        label = np.concatenate(labels, axis=0)
        log = {'loss': train_loss/num_examples,
               'pp_r2': pp_r2(pred_label, label),
               'mse': mse(pred_label, label),
               'rmse': rmse(pred_label, label),
               'mae': mae(pred_label, label),
               'pp_mse': pp_mse(pred_label, label).tolist(),
               'pp_rmse': pp_rmse(pred_label, label).tolist(),
               'pp_mae': pp_mae(pred_label, label).tolist(),
               }
        log['avg_r2'] = np.mean(log['pp_r2'])
        self.logger.write(log, epoch=epoch)
        self.save(os.path.join(self.exp_path, 'models', 'model.%d.t7'%epoch))
        return log

    def test_one_epoch(self, loader, epoch):
        self.G.eval()
        test_loss = 0.0
        num_examples = 0
        imgs = []
        pred_labels = []
        labels = []
        for data in tqdm(loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            pred_label = self.predict(img)
            loss = self.criterion(pred_label*self.param_scale, label*self.param_scale) \
                   + torch.sum((self.psf*(pred_label-label))**2)
            batch_size = img.size(0)
            test_loss += loss.item() * batch_size
            num_examples += batch_size
            imgs.append(img.cpu().numpy())
            labels.append(label.cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())
        img = np.concatenate(imgs, axis=0)
        label = np.concatenate(labels, axis=0)
        pred_label = np.concatenate(pred_labels, axis=0)
        log = {'loss': test_loss/num_examples,
               'img': img,
               'label': label,
               'pred_label': pred_label,
               'pp_r2': pp_r2(pred_label, label),
               'mse': mse(pred_label, label),
               'rmse': rmse(pred_label, label),
               'mae': mae(pred_label, label),
               'pp_mse': pp_mse(pred_label, label).tolist(),
               'pp_rmse': pp_rmse(pred_label, label).tolist(),
               'pp_mae': pp_mae(pred_label, label).tolist(),
               }
        log['avg_r2'] = np.mean(log['pp_r2'])
        self.logger.write(log, epoch=epoch, stage='test')
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.save(os.path.join(self.exp_path, 'models', 'model.best.t7'))
        return log

    def save(self, path):
#         if torch.cuda.device_count() > 1:
#             torch.save(self.G.module.state_dict(), path)
#         else:
        torch.save(self.G.state_dict(), path)

    def load(self, path):
        self.G.load_state_dict(torch.load(path))


class CGAN(Model):
    def __init__(self, args):
        super(CGAN, self).__init__(args)
        # self.D = D(args, emb_nn=resnet18()).to(self.device)
        self.D = SimpleD(args, emb_nn=None).to(self.device)
#         if torch.cuda.device_count() > 1:
#             self.D = nn.DataParallel(self.D)
        self.criterionGAN = GANLoss(args.gan_mode).to(self.device)
        self.criterion = nn.MSELoss()#nn.L1Loss()
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.lr)# betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.lr)# betas=(0.5, 0.999))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self, img, label, pred_label):
        pred_fake = self.D(img, pred_label.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.D(img, label)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, img, label, pred_label):
        pred_fake = self.D(img, pred_label)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterion(pred_label, label)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*100
        self.loss_G.backward()

    def optimize_parameters(self, img, label, pred_label):
        #update D
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D(img, label, pred_label)
        self.optimizer_D.step()

        #update G
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G(img, label, pred_label)
        self.optimizer_G.step()

    def train_one_epoch(self, loader, epoch):
        self.G.train()
        self.D.train()
        self.scheduler.step()
        train_loss = 0.0
        num_examples = 0
        pred_labels = []
        labels = []
        for data in tqdm(loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            pred_label = self.predict(img)
            self.optimize_parameters(img, label, pred_label)
            pred_labels.append(pred_label.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
            loss = self.criterion(pred_label, label)
            batch_size = img.size(0)
            train_loss += loss.item() * batch_size
            num_examples += batch_size
        pred_label = np.concatenate(pred_labels, axis=0)
        label = np.concatenate(labels, axis=0)
        log = {'loss': train_loss/num_examples,
               'pp_r2': pp_r2(pred_label, label),
               'mse': mse(pred_label, label),
               'rmse': rmse(pred_label, label),
               'mae': mae(pred_label, label),
               'pp_mse': pp_mse(pred_label, label).tolist(),
               'pp_rmse': pp_rmse(pred_label, label).tolist(),
               'pp_mae': pp_mae(pred_label, label).tolist(),
               }
        log['avg_r2'] = np.mean(log['pp_r2'])
        self.logger.write(log, epoch=epoch)
        self.save(os.path.join(self.exp_path, 'models', 'model.%d.t7'%epoch))
        return log

    def test_one_epoch(self, loader, epoch):
        self.G.eval()
        self.D.eval()
        test_loss = 0.0
        num_examples = 0
        imgs = []
        pred_labels = []
        labels = []
        for data in tqdm(loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            pred_label = self.predict(img)
            loss = self.criterion(pred_label, label)
            batch_size = img.size(0)
            test_loss += loss.item() * batch_size
            num_examples += batch_size
            imgs.append(img.cpu().numpy())
            labels.append(label.cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())
        img = np.concatenate(imgs, axis=0)
        label = np.concatenate(labels, axis=0)
        pred_label = np.concatenate(pred_labels, axis=0)
        log = {'loss': test_loss/num_examples,
               'img': img,
               'label': label,
               'pred_label': pred_label,
               'pp_r2': pp_r2(pred_label, label),
               'mse': mse(pred_label, label),
               'rmse': rmse(pred_label, label),
               'mae': mae(pred_label, label),
               'pp_mse': pp_mse(pred_label, label).tolist(),
               'pp_rmse': pp_rmse(pred_label, label).tolist(),
               'pp_mae': pp_mae(pred_label, label).tolist(),
               }
        log['avg_r2'] = np.mean(log['pp_r2'])
        self.logger.write(log, epoch=epoch, stage='test')
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.save(os.path.join(self.exp_path, 'models', 'model.best.t7'))
        return log


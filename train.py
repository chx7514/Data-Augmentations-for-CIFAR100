import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import errno
import os
import os.path as osp
import shutil
from collections import OrderedDict
import time
import argparse

from model.densenet import DenseNet121
from model.efficientnet import EfficientNetB0
from model.lenet import LeNet
from model.mobilenet import MobileNet
from model.regnet import RegNetX_200MF
from model.resnet import ResNet18
from model.resnext import ResNeXt29_2x64d
from model.shufflenet import ShuffleNetG2
from model.simpleDLA import SimpleDLA
from model.vgg import VGG_for_cifar100
from model.wideresnet import WideResNet_for_cifar100


model_options = ['lenet', 'vgg', 'resnet', 'resnext', 'wideresnet', 'regnet', 'densenet', 'mobilenet', 'efficientnet', 'simpleDLA', 'shufflenet']

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()

train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

train_dataset = CIFAR100(root='./cifar100', train=True, download=False, transform=train_transform)
valid_dataset = CIFAR100(root='./cifar100', train=False, download=False, transform=val_transform)

Batch_size = args.batch_size
train_loader = DataLoader(train_dataset,
                              batch_size=Batch_size,
                              shuffle=True,
                              num_workers=2)
valid_loader = DataLoader(valid_dataset,
                            batch_size=Batch_size,
                            num_workers=2)

if args.model == 'lenet':
    model = LeNet()
if args.model == 'vgg':
    model = VGG_for_cifar100()
if args.model == 'resnet':
    model = ResNet18()
if args.model == 'resnext':
    model = ResNeXt29_2x64d()
if args.model == 'wideresnet':
    model = WideResNet_for_cifar100()
if args.model == 'regnet':
    model = RegNetX_200MF()
if args.model == 'densenet':
    model = DenseNet121()
if args.model == 'mobilenet':
    model = MobileNet()
if args.model == 'efficientnet':
    model = EfficientNetB0()
if args.model == 'simpleDLA':
    model = SimpleDLA()
if args.model == 'shufflenet':
    model = ShuffleNetG2()

model = model.cuda()
torch.backends.cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile('./best_model_' + args.model + '.pth'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./best_model_' + args.model + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_acc = 0
    start_epoch = 0


path = os.path.join('./path/to/log/baseline', args.model)
writer = SummaryWriter(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    #滑动平均
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #topk准确率
    #预测结果前k个中出现的正确结果的次数
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mkdir_if_missing(directory):
    #创建文件夹，如果这个文件夹不存在的话
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best=False, fpath=''):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

def warp_tqdm(data_loader, disable_tqdm):
    #进度条打印
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, ncols=0)
    return tqdm_loader

def train(train_loader, model, criterion, optimizer, epoch):
    #每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):


        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=losses)
    return losses.avg, log

def test(test_loader, model):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for input, target in test_loader:

        # compute output
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output = model(input)

        # measure accuracy and record loss
        accs = accuracy(output.data, target, (1, 5))
        acc1, acc5 = accs[0], accs[1]
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time

    log = 'Test Acc@1: {top1.avg:.3f}\t Test Acc@5: {top5.avg:.3f}'.format(top1=top1, top5=top5)

    return top1.avg, top5.avg, log


num_epochs = args.epochs

train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=start_epoch-1)
criterion = nn.CrossEntropyLoss()
for epoch in range(start_epoch, num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    test_acc1, test_acc5, test_log = test(test_loader, model)
    train_acc1, train_acc5, _ = test(train_loader, model)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalars('top1 acc', {'train': train_acc1, 'test': test_acc1}, epoch)
    writer.add_scalars('top5 acc', {'train': train_acc5, 'test': test_acc5}, epoch)
    scheduler.step()
    log = train_log + test_log
    print(log)
    is_best = test_acc1 > best_acc
    best_acc = max(test_acc1, best_acc)
    if is_best:
        save_checkpoint({'epoch':epoch,
        'state_dict':model.state_dict(),
        'acc': best_acc,
        }, False, 'best_model_' + args.model + '.pth')

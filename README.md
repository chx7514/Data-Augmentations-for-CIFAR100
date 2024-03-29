# Data Augmentations for CIFAR-100

We implement eleven networks (see the model_options) and eight data augmentations (cutout, mixup, cutmix, random erasing, manifold mixup, autoaugment, randaugment and trivialaugment.

Run the train files to do the experiments. Dataset CIFAR-100 should be at `./cifar100`. The result is saved in tensorboard, and the path is like `./path/to/log/[Augmentation]/[Network]`

For example, to run the baseline experiment of ResNet18:

```
python train.py --model resnet
```

Run the mixup experiment of ResNet18:

```
python train_mixup.py --model resnet
```

Models should be chosen from below:

```
model_options = ['lenet', 'vgg', 'resnet', 'resnext', 'wideresnet', 'regnet', 
                 'densenet', 'mobilenet', 'efficientnet', 'simpleDLA', 'shufflenet']
```

It should be noted that for the manifold mixup, we only implement ResNet18 and WideResNet28-10.

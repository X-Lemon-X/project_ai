"""Decision networks: image -> (action logits over L / F / R, aux = log free distance ahead).

Input: float32 [N, 3, H, W] in [0, 1], RGB. Each model normalises with the ImageNet
mean / std itself, so an exported TorchScript model needs no preprocessing apart
from scaling the camera image to [0, 1].

    resnet18             pretrained ImageNet ResNet18, fine-tuned (main model)
    mobilenet_v3_small   pretrained, much lighter (for the real robot)
    efficientnet_b0      pretrained, stronger but slower
    smallcnn             5 conv blocks trained from scratch (baseline: what does pretraining add?)
"""

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Heads(nn.Module):
    def __init__(self, in_dim, num_classes=3, num_aux=1, hidden=128, dropout=0.3):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                    nn.Linear(hidden, num_classes))
        self.aux = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, num_aux))

    def forward(self, f):
        return self.action(f), self.aux(f)


class DecisionNet(nn.Module):
    def __init__(self, backbone, feat_dim, pretrained_backbone, num_classes=3, num_aux=1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone = backbone
        self.heads = Heads(feat_dim, num_classes, num_aux)
        self.pretrained_backbone = pretrained_backbone

    def features(self, x):
        return self.backbone((x - self.mean) / self.std)

    def forward(self, x):
        return self.heads(self.features(x))

    def param_groups(self, lr_backbone, lr_head, weight_decay):
        """Pretrained backbones get a smaller learning rate than the new heads."""
        lr_b = lr_backbone if self.pretrained_backbone else lr_head
        return [{"params": self.backbone.parameters(), "lr": lr_b, "weight_decay": weight_decay},
                {"params": self.heads.parameters(), "lr": lr_head, "weight_decay": weight_decay}]

    def set_backbone_trainable(self, on):
        for p in self.backbone.parameters():
            p.requires_grad_(on)


def _small_cnn():
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 192), block(192, 256),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return net, 256


def build_model(name="resnet18", pretrained=True, num_classes=3, num_aux=1):
    tv = torchvision.models
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        dim, m.fc = m.fc.in_features, nn.Identity()
    elif name == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        dim, m.classifier = m.classifier[0].in_features, nn.Identity()
    elif name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        dim, m.classifier = m.classifier[1].in_features, nn.Identity()
    elif name == "smallcnn":
        (m, dim), pretrained = _small_cnn(), False
    else:
        raise ValueError(f"unknown model {name!r}")
    return DecisionNet(m, dim, pretrained, num_classes, num_aux)


MODELS = ("resnet18", "mobilenet_v3_small", "efficientnet_b0", "smallcnn")

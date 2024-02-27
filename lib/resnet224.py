import torch
import torch.nn as nn
import torchvision


class ResNet34(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        self.n_features = out_size
        # del self.resnet34.fc

        for param in self.resnet34.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(nn.Linear(self.resnet34.fc.in_features, out_size))

    def forward(self, x):
        # Hemmer et al ===
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        x = self.resnet34.maxpool(x)
        x = self.resnet34.layer1(x)
        x = self.resnet34.layer2(x)
        x = self.resnet34.layer3(x)
        x = self.resnet34.layer4(x)
        x = self.resnet34.avgpool(x)

        x = self.classifier(torch.flatten(x, 1))
        return x

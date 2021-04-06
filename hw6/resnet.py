
import torch.nn as nn
#add imports as necessary

class ResNet:

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(64) # CHECK LATER
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        #self.layer1 = self.new_block(64, 64, 1)
        #self.layer2 = self.new_block(64, 128, 1)
        #self.layer3 = self.new_block(128, 256, 1)
        #self.layer4 = self.new_block(256, 512, 1)
        self.layer1 = nn.Conv2d(64, 64, 3)
        self.layer2 = nn.Conv2d(64, 128, 3)
        self.layer3 = nn.Conv2d(128, 256, 3)
        self.layer4 = nn.Conv2d(256, 512, 3)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax()


    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x_copy = x.clone().detach()
        x = self.layer1(x)
        x = self.layer1(x)
        x += x_copy
        x_copy = x.clone().detach()
        x = self.layer1(x)
        x = self.layer1(x)
        x += x_copy

        x = self.layer2(x)
        x = self.layer2(x)
        x_copy = x.clone().detach()
        x = self.layer2(x)
        x = self.layer2(x)
        x += x_copy

        x = self.layer3(x)
        x = self.layer3(x)
        x_copy = x.clone().detach()
        x = self.layer3(x)
        x = self.layer3(x)
        x += x_copy

        x = self.layer4(x)
        x = self.layer4(x)
        x_copy = x.clone().detach()
        x = self.layer4(x)
        x = self.layer4(x)
        x += x_copy

        x = self.avgpool(x)
        x = x.reshape(512, 1)
        x = self.fc(x)
        x = self.softmax()
        return x



    def new_block(self, in_planes, out_planes, stride):
        layers = []
        #TODO: make a convolution with the above params
        layers.append(nn.Conv2d(in_planes, out_planes, 3, stride=stride))
        layers.append(nn.Conv2d(in_planes, out_planes, 3, stride=stride))
        layers.append(nn.Conv2d(in_planes, out_planes, 3, stride=stride))
        layers.append(nn.Conv2d(in_planes, out_planes, 3, stride=stride))
        return nn.Sequential(*layers)
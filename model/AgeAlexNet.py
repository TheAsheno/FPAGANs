import torch.nn as nn
import torch
import os
import sys
sys.path.append("E:\\FaceNN\\FPAGAN")
from utils.network import Conv2d, load_params
from utils.io import Img_to_zero_center
import torchvision
from PIL import Image

class AgeAlexNet(nn.Module):
    def __init__(self, modelpath=None):
        super(AgeAlexNet, self).__init__()

        self.features = nn.Sequential(
            Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2,2e-5,0.75),

            Conv2d(96, 256, kernel_size=5, stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2, 2e-5, 0.75),

            Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            Conv2d(384, 384, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),

            Conv2d(384, 256, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.age_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )
        self.loss = nn.CrossEntropyLoss().cuda()
        if modelpath is not None:
            load_params(self, modelpath)

        self.Conv3_feature_module = nn.Sequential()
        self.Conv4_feature_module = nn.Sequential()
        self.Conv5_feature_module = nn.Sequential()
        self.Pool5_feature_module = nn.Sequential()
        for x in range(10):
            self.Conv3_feature_module.add_module(str(x), self.features[x])
        for x in range(10,12):
            self.Conv4_feature_module.add_module(str(x),self.features[x])
        for x in range(12,14):
            self.Conv5_feature_module.add_module(str(x),self.features[x])
        for x in range(14,15):
            self.Pool5_feature_module.add_module(str(x),self.features[x])

    def forward(self, x):
        self.conv3_feature = self.Conv3_feature_module(x)
        self.conv4_feature = self.Conv4_feature_module(self.conv3_feature)
        self.conv5_feature = self.Conv5_feature_module(self.conv4_feature)
        pool5_feature = self.Pool5_feature_module(self.conv5_feature)
        self.pool5_feature = pool5_feature
        flattened = pool5_feature.view(pool5_feature.size(0), -1)
        return self.age_classifier(flattened)

    def save_model(self, dir, filename):
        torch.save(self.state_dict(), os.path.join(dir, filename))

if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((227, 227)),
        torchvision.transforms.ToTensor(),
        Img_to_zero_center()
    ])
    model = AgeAlexNet(modelpath='checkpoint\\AgeClassify\\saved_parameters\\epoch_20_iter_0.pth').cuda()
    img = Image.open("dataset\\CACD2000\\image\\expriment\\30_Natalie_Portman_0015.jpg")
    img = transforms(img).unsqueeze(0)
    label = torch.tensor([2])
    model.eval()
    with torch.no_grad():
        img = img.cuda()
        label = label.cuda()
        output = model(img).argmax(dim=1)
        print("Output:", output.item())
        print("Real:", label.item())
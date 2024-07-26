import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_planes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_planes=in_planes)


model = ResNet18()
# Download the model
#wget -O resnet18_demo.pth http://download.huan-zhang.com/models/auto_lirpa/resnet18_natural.pth
# Load pretrained weights. This pretrained model is for illustration only; it does not represent state-of-the-art classification performance.
checkpoint = torch.load("resnet18_demo.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_data = datasets.CIFAR10(
    "./data", train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]))
# Choose one image from the dataset.
idx = 123
image = test_data[idx][0].view(1,3,32,32)
label = data = test_data[idx][1]
print('Ground-truth label:', label)
print('Model prediction:', model(image))


# Step 1: wrap model with BoundedModule. The "conv_mode" option enables efficient CNN bounds on GPUs.
bounded_model = BoundedModule(model, torch.zeros_like(image), bound_opts={"conv_mode": "patches"})
bounded_model.eval()

# Step 2: define perturbation. Here we use a Linf perturbation on input image.
eps = 0.003
norm = np.inf
ptb = PerturbationLpNorm(norm = norm, eps = eps)
# Input tensor is wrapped in a BoundedTensor object.
bounded_image = BoundedTensor(image, ptb)
# We can use BoundedTensor to get model prediction as usual. Regular forward/backward propagation is unaffected.
print('Model prediction:', bounded_model(bounded_image))


# Step 3: compute bounds using the compute_bounds() method.
print('Bounding method: backward (CROWN, DeepPoly)')
with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
  lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method='CROWN')

# Auxillary function to print bounds.
def print_bounds(lb, ub):
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()
    for j in range(10):
        print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(
            j=j, l=lb[0][j], u=ub[0][j], r=ub[0][j] - lb[0][j]))

print_bounds(lb, ub)


# Our library also supports the interval bound propagation (IBP) based bounds,
# but it produces much looser bounds.
print('Bounding method: IBP')
with torch.no_grad():
  lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method='IBP')

print_bounds(lb, ub)

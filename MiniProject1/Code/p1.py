import torch
import torch.nn as nn
import torch.nn.functional as F
import ssl
import torchvision
from torchvision import transforms
from timm.loss import LabelSmoothingCrossEntropy
import logging
import argparse
import os

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
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

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=128, metavar='N', type=int)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--decay_step', default=50, metavar='N', type=int)
parser.add_argument('--checkpoint', default='resnet-18', type=str, metavar='checkpoint')
parser.add_argument('--smooth', action='store_true',default=True)
args = parser.parse_args()
print(args)

if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
    print("Checkpoint created! Checkpoint will be saved at %s" % args.checkpoint)
else:
    print("checkpoint will be saved at %s" % args.checkpoint)

logging.basicConfig(filename="{}/resnet-18.log".format(args.checkpoint), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

trfl = [transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
]
tefl = [transforms.ToTensor()]
trfl = transforms.Compose(trfl)
tefl = transforms.Compose(tefl)
interval = 50

trainingdata = torchvision.datasets.CIFAR10('./CIFAR10/',
                                            train=True,
                                            download=True,
                                            transform=trfl
                                            )
testdata = torchvision.datasets.CIFAR10('./CIFAR10/',train=False,download=True,transform=tefl)

net = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()
if args.smooth:
    print("using label smooth")
    Loss = LabelSmoothingCrossEntropy(0.1)
else:
    print("using cross entropy loss")
    Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step)

trainDataLoader = torch.utils.data.DataLoader(trainingdata,batch_size=args.bs,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testdata,batch_size=args.bs,shuffle=False)
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

for epoch in range(200):
    train_loss = 0.0
    test_loss = 0.0
    correct_points_train = 0
    correct_points_test = 0

    for i, data in enumerate(trainDataLoader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        predicted_output = net(images)
        fit = Loss(predicted_output,labels)
        fit.backward()
        optimizer.step()
        train_loss += fit.item()
        correct_points_train += (torch.eq(torch.max(predicted_output, 1)[1],labels).sum()).data.cpu().numpy()
        if i % interval == 0 and i != 0:
            info = 'Epoch [%d][%d/%d] lr: %f, loss_ce: %f'%(epoch, i, len(trainDataLoader),
                                                                  scheduler.get_last_lr()[0], fit.item())
            print(info)
            logger.info(info)

    scheduler.step()
    for data in testDataLoader:
        with torch.no_grad():
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            predicted_output = net(images)
            fit = Loss(predicted_output,labels)
            test_loss += fit.item()
            correct_points_test += (torch.eq(torch.max(predicted_output, 1)[1],labels).sum()).data.cpu().numpy()
    train_loss = train_loss/len(trainDataLoader)
    test_loss = test_loss/len(testDataLoader)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_acc = correct_points_train*100/len(trainingdata)
    test_acc = correct_points_test*100/len(testdata)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    info = '==== Epoch %s, Train loss %f, Test acc %f ====' % (epoch, train_loss, test_acc)
    print(info)
    logger.info(info)
    if epoch % interval == 0 and epoch != 0:
        info = 'saved checkpoint at epoch {}'.format(epoch)
        print(info)
        logger.info(info)
        torch.save(net.state_dict(), '{}/resnet-18-{}.pth'.format(args.checkpoint, epoch))

torch.save(net.state_dict(), '{}/resnet-18-final.pth'.format(args.checkpoint))
from __future__ import print_function
import argparse
import torch
from torch import nn, optim, cuda
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

batch_size = 256 # we use SGD with a mini-batch size of 256
transform = transforms.Compose([
    transforms.Pad(4), # Pad the given PIL Image on all sides with the given “pad” value.
    transforms.RandomHorizontalFlip(), # A crop is randomly sampled from an image or its horizontal flip
    transforms.RandomCrop(32),
    transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data',
                              train=True,
                              transform=transform,
                              download=True)
test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.convolution1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(out_channels)
    self.convolution2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(out_channels)
    if stride != 1:
      self.downsampling = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    else:
      self.downsampling = None
  
  def forward(self, x):
    residual = x
    out = self.convolution1(x)
    out = self.batchnorm1(out)
    out = F.relu(out)
    out = self.convolution2(out)
    out = self.batchnorm2(out)
    if self.downsampling:
      residual = self.downsampling(x) # if stride==2, 원래의 값의 가로세로를 반으로 줄여줘야 하므로 kernel_size=1, stride=2인 conv2d를 통과시켜준다..
    out += residual
    return out
  
class ResNet(nn.Module):
  def __init__(self, blocks_per_layer=5):
    super(ResNet, self).__init__()
    # input: 32 * 32 image <=> 3 * 32 * 32
    self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    self.batchnorm = nn.BatchNorm2d(16)
    # 현재: 16개의 32 * 32
    self.layer_2n = self.make_layer(16, 16, 1, blocks_per_layer)
    # 현재: 16개의 32 * 32
    self.layer_4n = self.make_layer(16, 32, 2, blocks_per_layer)
    # 현재: 32개의 16 * 16
    self.layer_6n = self.make_layer(32, 64, 2, blocks_per_layer)
    # 현재: 64개의 8 * 8
    self.avg_pool = nn.AvgPool2d(8, stride=1)
    # cifar-10은 10개의 class에 따른 이미지를 데이터셋으로 지닌다.
    self.fc = nn.Linear(64, 10)

  def make_layer(self, in_channels, out_channels, stride, blocks_per_layer):
    layers = []
    # first block
    layers.append(ResidualBlock(in_channels, out_channels, stride))
    # 나머지 n-1 block
    for _ in range(1, blocks_per_layer):
      layers.append(ResidualBlock(out_channels, out_channels, 1))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv(x)
    x = self.batchnorm(x)
    x = F.relu(x)

    x = self.layer_2n(x)
    x = self.layer_4n(x)
    x = self.layer_6n(x)
    # The networks ends with a global average pooling
    x = self.avg_pool(x)
    # a 10-way fully connected layer
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    # and softmax
    x = F.log_softmax(x)
    return x
    

model = ResNet(18).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) 
# we use SGD with a mini-batch size of 256 
# the learning rate starts from 0.1 and is divided by 10 when the error plateaus(안정)
# we use a weight decay of 0.0001 and a momentum of 0.9

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 기울기 초기화
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # back propagation, 역전파
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 출처: https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840/4
# INPUTS: output은 [batch_size, category_count]의 모양을 띈다. (== batch_size * category_count 크기의 2차원 배열이라 생각하자)
# target: [batch_size] * category_count인데 이때 하나의 값만 1인, 즉 one-hot encoding이 되어있다.
# topk: 구하고 싶은 top-k error에 해당하는 k
def correct_count(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
   # 이 경우 gradient calculation이 필요없으므로 꺼주자.
    with torch.no_grad():
        maxk = topk
    # topk 함수는 output에서 dimth dimesion에 있는 배열에서 가장 큰 값을 반환한다.
    # output이 [batch_size, category_count], dim=1 이므로 각 batch에 대해 maxk만큼 큰 값을 반환해줄거다.
    # input=maxk, so we will select maxk number of classes 
    # so result will be [batch_size,maxk]
    # topk의 반환값은 tuple (values, indexes) 꼴이다.
    # 여기서는 실제 target class에 해당하는지 확인하기 위해 index만 필요하므로, 다음과 같은 code를 구성한다.
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    # pred tensor를 [maxk, batch_size]꼴로 transpose 시켜준다.
        pred = pred.t()
    # we flatten target and then expand target to be like pred 
    # target [batch_size] becomes [1,batch_size]
    # target [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times. 
    # when you compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size] filled with 1 and 0 for correct and wrong class assignments
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    # 예를들어.. 
    # correct=([[0, 0, 1,  ..., 0, 0, 0],
    #          [1, 0, 0,  ..., 0, 0, 0],
    #          [0, 0, 0,  ..., 1, 0, 0],
    #          [0, 0, 0,  ..., 0, 0, 0],
    #          [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8)"""
    # correct배열을 dimesion1로 압축하고, float 선언한뒤 값을 다 더하자
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        return correct_k.item()

def test():
    model.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        pred_1 = output.data.max(1, keepdim=True)[1]
        correct_1 += correct_count(output, target, 1)
        correct_5 += correct_count(output, target, 5)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, top-1 Error: {}/{} ({:.0f}%)'.format(
        test_loss, len(test_loader.dataset)-correct_1, len(test_loader.dataset),
        100. * (len(test_loader.dataset)-correct_1) / len(test_loader.dataset)))
    print('Test set: Average loss: {:.4f}, top-5 Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, len(test_loader.dataset)-correct_5, len(test_loader.dataset),
        100. * (len(test_loader.dataset)-correct_5) / len(test_loader.dataset)))


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 80):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
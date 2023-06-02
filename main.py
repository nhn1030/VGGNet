import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from model import VGGNet16
from data_loader import get_train_loader
from data_loader import get_test_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VGGNet16()

root = 'datasets'  # 실제 데이터셋이 위치한 디렉토리 경로로 수정해주세요.
batch_size = 64
num_workers = 2

train_loader = get_train_loader(root, batch_size, num_workers)
test_loader = get_test_loader(root, batch_size, num_workers)


learning_rate = 0.001
file_name = 'VGGNet16_CIFAR10.pth'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)

        
def train(epoch):
        print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            current_correct = predicted.eq(targets).sum().item()
            correct += current_correct
            
            if batch_idx % 100 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current batch average train accuracy:', current_correct / targets.size(0))
                print('Current batch average train loss:', loss.item() / targets.size(0))

        print('\nTotal average train accuarcy:', correct / total)
        print('Total average train loss:', train_loss / total)


def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('\nTotal average test accuarcy:', correct / total)
    print('Total average test loss:', loss / total)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

start_time = time.time()

for epoch in range(0, 150):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
    print('\nTime elapsed:', time.time() - start_time)
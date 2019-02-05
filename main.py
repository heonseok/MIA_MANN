import argparse
import sys
import os
import csv

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *

import torchvision
import torchvision.transforms as transforms

from model import *
from utils import *


def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='nn')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_hidden', type=int, default=50)
parser.add_argument('--epochs', type=int, default=2)

parser.add_argument('--init_indices', type=str2bool, default='f')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--idx_dir', type=str, default='./indices')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--mia_data_dir', type=str, default='./mia_dataset')

parser.add_argument('--shadow_train', type=str2bool, default='f')
parser.add_argument('--target_train', type=str2bool, default='f')
parser.add_argument('--build_mia_data', type=str2bool, default='f')
parser.add_argument('--attack_train', type=str2bool, default='f')
parser.add_argument('--attack_test', type=str2bool, default='t')
# parser.add_argument('--shadow_test', type=str2bool, default='t')

config = parser.parse_args()
print(config)

if not os.path.exists(config.idx_dir):
    os.mkdir(config.idx_dir)
if not os.path.exists(config.model_dir):
    os.mkdir(config.model_dir)
if not os.path.exists(config.mia_data_dir):
    os.mkdir(config.mia_data_dir)

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def get_shadow_name(idx):
    return 'shadow_{:0>3d}'.format(idx)


def concat_train_test():
    # merge original train and test to one dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    total_set = ConcatDataset([train_set, test_set])
    total_size = total_set.__len__()
    print('Total set size: ' + str(total_size))

    return total_set


def init_indices(size):
    permu_idx = np.random.permutation(size)
    half_size = int(size/2)

    target_idx = permu_idx[:half_size]
    shadow_idx = permu_idx[half_size:]

    write_train_test_indices(target_idx, 'target', config.idx_dir)

    for k_idx in range(config.K):
        write_train_test_indices(shadow_idx, get_shadow_name(k_idx), config.idx_dir)


def write_train_test_indices(idx_list, name):
    print('size of data : ' + str(len(idx_list)))
    permu_idx = np.random.permutation(len(idx_list))
    train_idx = permu_idx[:int(len(idx_list)/2)]
    test_idx = permu_idx[int(len(idx_list)/2):]

    f = open(os.path.join(config.idx_dir, '{}.csv'.format(name)), 'w')
    wr = csv.writer(f)
    wr.writerow(idx_list[train_idx])
    wr.writerow(idx_list[test_idx])
    f.close()


def load_train_test_indices(name):
    df = pd.read_csv(os.path.join(config.idx_dir, '{}.csv'.format(name)), header=None)
    indices = df.values
    return indices[0], indices[1]


def get_loaders(total_set, name):
    train_idx, test_idx = load_train_test_indices(name)
    train_loader = torch.utils.data.DataLoader(total_set, batch_size=config.batch_size,
                                               shuffle=False, sampler=SubsetRandomSampler(train_idx))
    test_loader = torch.utils.data.DataLoader(total_set, batch_size=config.batch_size,
                                              shuffle=False, sampler=SubsetRandomSampler(test_idx))

    return train_loader, test_loader


def base_train(net, name, train_loader):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(config.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(net.state_dict(), os.path.join(config.model_dir, name + '.pth'))


def base_test(net, name, test_loader):
    net.load_state_dict(torch.load(os.path.join(config.model_dir, name + '.pth')))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\tAccuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def save_prediction(net, name, data_loader, train_flag):

    if train_flag == 1:
        in_out_flag = np.ones([config.batch_size, 1])
    else:
        in_out_flag = np.zeros([config.batch_size, 1])

    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            # print(images)
            outputs = net(images).cpu().numpy().squeeze().reshape(config.batch_size, -1)

            labels = labels.cpu().numpy().reshape(config.batch_size, -1)

            if idx == 0:
                mia_data = np.hstack([outputs, labels, in_out_flag])
            else:
                mia_data = np.vstack([mia_data, np.hstack([outputs, labels, in_out_flag])])

    df = pd.DataFrame(mia_data)
    df.to_csv(os.path.join(config.mia_data_dir, '{}.csv'.format(name)), header=False, index=False)


def attack_train(net, train_in_loader, train_out_loader):
    net.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(config.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (in_data, out_data) in enumerate(zip(train_in_loader, train_out_loader)):
            x1 = torch.from_numpy(np.vstack([in_data[0], out_data[0]]))
            label = torch.from_numpy(np.vstack([in_data[2].reshape([-1, 1]), out_data[2].reshape([-1, 1])])).float()

            x2 = np.vstack([in_data[1].reshape([-1, 1]), out_data[1].reshape([-1, 1])])
            onehot_x2 = np.zeros((config.batch_size, 10), dtype=float)
            for idx in range(config.batch_size):
                onehot_x2[idx, int(x2[idx])] = 1

            x2 = torch.from_numpy(onehot_x2).float()
            # print(x2)
            # get the inputs
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x1, x2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(net.state_dict(), os.path.join(config.model_dir, 'attack.pth'))


def attack_test(net, in_loader, out_loader):
    net.load_state_dict(torch.load(os.path.join(config.model_dir, 'attack.pth')))
    net.eval()

    with torch.no_grad():
        for i, (in_data, out_data) in enumerate(zip(in_loader, out_loader)):
            x1 = torch.from_numpy(np.vstack([in_data[0], out_data[0]]))
            label = torch.from_numpy(np.vstack([in_data[2].reshape([-1, 1]), out_data[2].reshape([-1, 1])]))

            x2 = np.vstack([in_data[1].reshape([-1, 1]), out_data[1].reshape([-1, 1])])
            onehot_x2 = np.zeros((config.batch_size, 10))
            for idx in range(config.batch_size):
                onehot_x2[idx, int(x2[idx])] = 1

            x2 = torch.from_numpy(onehot_x2).float()
            # get the inputs
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            # forward + backward + optimize
            outputs = net(x1, x2)

            if i == 0:
                target = label.cpu()
                predict = outputs.cpu()
            else:
                target = np.vstack([target, label.cpu()])
                predict = np.vstack([predict, outputs.cpu()])

            # print(outputs)
            # print(label)

    print(target.shape)
    print(predict.shape)
    auc, acc = calculate_auc_acc(target, predict)
    print(auc, acc)

    sys.exit(1)


    print('\tAccuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def merge_mia_sets(file_name_list, in_out_label):
    if in_out_label == 1:
        total_name = 'shadow_total_in.csv'
    else:
        total_name = 'shadow_total_out.csv'

    with open(os.path.join(config.mia_data_dir, total_name), 'w') as outfile:
        for file_name in file_name_list:
            with open(os.path.join(config.mia_data_dir, file_name)) as infile:
                for line in infile:
                    outfile.write(line)


def main():
    ###################################################
    # 1. merge original train/test sets
    # 2. save indices for target(1)/shadow(K) sets
    # 3. train/test target model (using base_train/test) todo
    # 4. train/test shadow model (using base_train/test) todo
    # 5. save mia_data (prediction_vec, class, in/out)
    # 6. train/test attack model (using attck_train/test) todo
    ###################################################

    total_set = concat_train_test()

    if config.init_indices:
        print('Init indices for target/shadow')
        init_indices(total_set.__len__())

    if config.target_train:
        print('Training target model')
        net = Net()
        net.to(device)
        train_loader, test_loader = get_loaders(total_set, 'target')
        base_train(net, 'target', train_loader)
        base_test(net, 'target', test_loader)

    if config.shadow_train:
        print('Training shadow model')
        for shadow_idx in range(config.K):
            net = Net()
            net.to(device)

            shadow_name = get_shadow_name(shadow_idx)
            print(shadow_name)

            train_loader, test_loader = get_loaders(total_set, shadow_name)
            base_train(net, shadow_name, train_loader)
            base_test(net, shadow_name, test_loader)

    if config.build_mia_data:
        print('Building mia dataset')

        # For target model
        net = Net()
        net.to(device)
        train_loader, test_loader = get_loaders(total_set, 'target')

        net.load_state_dict(torch.load(os.path.join(config.model_dir, 'target.pth')))
        net.eval()

        print('target train')
        save_prediction(net, 'target_in', train_loader, 1)

        print('target test')
        save_prediction(net, 'target_out', test_loader, 0)

        # For shadow model
        in_file_name_list =[]
        out_file_name_list =[]

        for shadow_idx in range(config.K):
            net = Net()
            net.to(device)

            shadow_name = get_shadow_name(shadow_idx)
            train_loader, test_loader = get_loaders(total_set, shadow_name)

            net.load_state_dict(torch.load(os.path.join(config.model_dir, shadow_name + '.pth')))
            net.eval()

            print(shadow_name, 'train')
            save_prediction(net, shadow_name + '_in', train_loader, 1)
            in_file_name_list.append(shadow_name + '_in.csv')

            print(shadow_name, 'test')
            save_prediction(net, shadow_name + '_out', test_loader, 0)
            out_file_name_list.append(shadow_name + '_out.csv')

        merge_mia_sets(in_file_name_list, 1)
        merge_mia_sets(out_file_name_list, 0)

    if config.attack_train:
        print('Training attack model')

        name = 'shadow_in.csv'
        mia_set_in = MIADataset(name, config.mia_data_dir)

        name = 'shadow_out.csv'
        mia_set_out = MIADataset(name, config.mia_data_dir)

        in_loader = torch.utils.data.DataLoader(mia_set_in, batch_size=int(config.batch_size/2), shuffle=True)
        out_loader = torch.utils.data.DataLoader(mia_set_out, batch_size=int(config.batch_size/2), shuffle=True)

        attack_model = AttackModel()
        attack_model.to(device)
        attack_train(attack_model, in_loader, out_loader)

    if config.attack_test:
        print('Testing attack model')

        name = 'target_in.csv'
        mia_set_in = MIADataset(name, config.mia_data_dir)

        name = 'target_out.csv'
        mia_set_out = MIADataset(name, config.mia_data_dir)

        in_loader = torch.utils.data.DataLoader(mia_set_in, batch_size=int(config.batch_size/2), shuffle=True)
        out_loader = torch.utils.data.DataLoader(mia_set_out, batch_size=int(config.batch_size/2), shuffle=True)

        attack_model = AttackModel()
        attack_model.to(device)
        attack_test(attack_model, in_loader, out_loader)


if __name__ == "__main__":
    main()

import argparse
import os
import multiprocessing

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from data import classification_dataset
from models import mobile_net_v2_model


def train(train_input_dir_path, valid_input_dir_path, classes_txt_path, epochs, step_size, batch_size,
          test_max_sample, image_size, output_dir_path):
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    # train dataset
    train_dataset = classification_dataset.MNISTImageDataset(train_input_dir_path, classes, step_size, image_size,
                                                             T.Compose([T.RandomRotation(degrees=[-7.5, 7.5]),
                                                                        T.RandomErasing(p=0.1)]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=multiprocessing.cpu_count() // 4)

    # test dataset
    test_dataset = classification_dataset.MNISTImageDataset(valid_input_dir_path, classes, test_max_sample, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=multiprocessing.cpu_count() // 4)

    # prepare model
    model = mobile_net_v2_model.MobileNetV2Model(len(classes))

    # prepare loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # train
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pth')
    parser.add_argument('--train_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/train')
    parser.add_argument('--valid_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/valid')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-mnist-classification-dataset/classes.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--output_dir_path', type=str, default='~/output_model')
    args = parser.parse_args()

    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.valid_input_dir_path = os.path.expanduser(args.valid_input_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)
    train(args.train_input_dir_path, args.valid_input_dir_path, args.classes_txt_path,
          args.epochs, args.step_size, args.batch_size, args.test_max_sample,
          (args.image_height, args.image_width), args.output_dir_path)
